import Accelerate
import Foundation
import OSLog
import os.signpost

/// K-Means clustering for forcing a specific number of speakers.
@available(macOS 14.0, iOS 17.0, *)
struct KMeansClustering {

    private static let logger = AppLogger(category: "KMeansClustering")

    /// GuLP fork: the seed the diarizer's speaker-count re-clustering uses, so the
    /// same audio always splits the same way (pyannote seeds its K-Means with 42 too).
    static let defaultSeed: UInt64 = 42
    /// GuLP fork: k-means++ restarts for that re-clustering.
    static let defaultRestarts = 10
    private static let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )

    /// Clusters embeddings into exactly `numClusters` groups.
    ///
    /// - Parameters:
    ///   - embeddings: Array of embedding vectors (each is a [Double] of same dimension).
    ///   - numClusters: Target number of clusters.
    ///   - maxIterations: Maximum iterations before stopping.
    ///   - seed: Optional seed for reproducible centroid initialization.
    /// - Returns: Cluster assignment for each embedding (0-indexed).
    static func cluster(
        embeddings: [[Double]],
        numClusters: Int,
        maxIterations: Int = 300,
        seed: UInt64? = nil
    ) -> [Int] {
        clusterWithCentroids(
            embeddings: embeddings,
            numClusters: numClusters,
            maxIterations: maxIterations,
            seed: seed
        ).clusters
    }

    /// GuLP fork: k-means++ starts, `restarts` times, keeping the run with the
    /// lowest inertia (sklearn's `n_init`, which pyannote uses for the same job).
    /// One random start can settle on a split that has nothing to do with who is
    /// speaking.
    static func clusterWithCentroids(
        embeddings: [[Double]],
        numClusters: Int,
        maxIterations: Int = 300,
        seed: UInt64,
        restarts: Int
    ) -> (clusters: [Int], centroids: [[Double]]) {
        guard restarts > 1, embeddings.count > numClusters, numClusters > 0,
            let dimension = embeddings.first?.count, dimension > 0
        else {
            return clusterWithCentroids(
                embeddings: embeddings, numClusters: numClusters, maxIterations: maxIterations, seed: seed)
        }
        let normalized = normalizeEmbeddings(embeddings)
        var rng = SeededRNG(seed: seed)
        var best: (clusters: [Int], centroids: [[Double]], inertia: Double)?
        for _ in 0..<restarts {
            var centroids = plusPlusCentroids(from: normalized, k: numClusters, rng: &rng)
            var assignments = [Int](repeating: -1, count: normalized.count)
            for _ in 0..<maxIterations {
                let next = assignToCentroids(normalized, centroids: centroids)
                if next == assignments { break }
                assignments = next
                centroids = updateCentroids(
                    embeddings: normalized, assignments: assignments, k: numClusters, dimension: dimension, rng: &rng)
            }
            let inertia = zip(normalized, assignments).reduce(0.0) { total, pair in
                total + euclideanDistanceSquared(pair.0, centroids[pair.1])
            }
            if best == nil || inertia < best!.inertia {
                best = (assignments, centroids, inertia)
            }
        }
        return (best!.clusters, best!.centroids)
    }

    /// k-means++: each next start is drawn in proportion to its squared distance
    /// from the starts chosen so far.
    private static func plusPlusCentroids(from embeddings: [[Double]], k: Int, rng: inout SeededRNG) -> [[Double]] {
        var centroids = [embeddings[Int(rng.next() % UInt64(embeddings.count))]]
        while centroids.count < k {
            let distances = embeddings.map { embedding in
                centroids.map { euclideanDistanceSquared(embedding, $0) }.min() ?? 0
            }
            let total = distances.reduce(0, +)
            guard total > 0 else { break }
            var target = Double(rng.next() % 1_000_000_007) / 1_000_000_007.0 * total
            var chosen = distances.count - 1
            for (index, distance) in distances.enumerated() {
                target -= distance
                if target <= 0 {
                    chosen = index
                    break
                }
            }
            centroids.append(embeddings[chosen])
        }
        return centroids
    }

    /// Clusters embeddings and returns both assignments and centroids.
    static func clusterWithCentroids(
        embeddings: [[Double]],
        numClusters: Int,
        maxIterations: Int = 300,
        seed: UInt64? = nil
    ) -> (clusters: [Int], centroids: [[Double]]) {
        let kmeansState = signposter.beginInterval("KMeans Clustering")
        defer { signposter.endInterval("KMeans Clustering", kmeansState) }

        let count = embeddings.count
        guard count > 0 else {
            return ([], [])
        }
        guard let dimension = embeddings.first?.count, dimension > 0 else {
            return (Array(repeating: 0, count: count), [])
        }

        let k = min(numClusters, count)
        if k <= 0 {
            return (Array(repeating: 0, count: count), [])
        }
        if count <= k {
            return (Array(0..<count), embeddings)
        }

        var rng = SeededRNG(seed: seed ?? UInt64.random(in: 0...UInt64.max))
        let normalized = normalizeEmbeddings(embeddings)
        var centroids = initializeCentroids(from: normalized, k: k, rng: &rng)
        var assignments = [Int](repeating: 0, count: count)

        for iteration in 0..<maxIterations {
            let newAssignments = assignToCentroids(normalized, centroids: centroids)
            if newAssignments == assignments {
                assignments = newAssignments
                break
            }
            assignments = newAssignments
            centroids = updateCentroids(
                embeddings: normalized,
                assignments: assignments,
                k: k,
                dimension: dimension,
                rng: &rng
            )

            if iteration > 80 {
                logger.warning("K-Means slow convergence at iteration \(iteration + 1)")
            }
        }

        return (assignments, centroids)
    }

    private static func normalizeEmbeddings(_ embeddings: [[Double]]) -> [[Double]] {
        embeddings.map { embedding in
            var norm: Double = 0
            vDSP_svesqD(embedding, 1, &norm, vDSP_Length(embedding.count))
            norm = sqrt(norm)
            guard norm > 1e-10 else { return embedding }
            var invNorm = 1.0 / norm
            var result = [Double](repeating: 0, count: embedding.count)
            vDSP_vsmulD(embedding, 1, &invNorm, &result, 1, vDSP_Length(embedding.count))
            return result
        }
    }

    private static func initializeCentroids(
        from embeddings: [[Double]],
        k: Int,
        rng: inout SeededRNG
    ) -> [[Double]] {
        var indices = Array(embeddings.indices)
        indices.shuffle(using: &rng)
        return Array(indices.prefix(k).map { embeddings[$0] })
    }

    private static func assignToCentroids(_ embeddings: [[Double]], centroids: [[Double]]) -> [Int] {
        embeddings.map { embedding in
            var bestCluster = 0
            var bestDistance = Double.greatestFiniteMagnitude
            for (idx, centroid) in centroids.enumerated() {
                let dist = euclideanDistanceSquared(embedding, centroid)
                if dist < bestDistance {
                    bestDistance = dist
                    bestCluster = idx
                }
            }
            return bestCluster
        }
    }

    private static func euclideanDistanceSquared(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return Double.greatestFiniteMagnitude }
        var diff = [Double](repeating: 0, count: a.count)
        vDSP_vsubD(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var result: Double = 0
        vDSP_svesqD(diff, 1, &result, vDSP_Length(a.count))
        return result
    }

    private static func updateCentroids(
        embeddings: [[Double]],
        assignments: [Int],
        k: Int,
        dimension: Int,
        rng: inout SeededRNG
    ) -> [[Double]] {
        var sums = [[Double]](repeating: [Double](repeating: 0, count: dimension), count: k)
        var counts = [Int](repeating: 0, count: k)

        for (idx, cluster) in assignments.enumerated() {
            let embedding = embeddings[idx]
            counts[cluster] += 1
            for d in 0..<dimension {
                sums[cluster][d] += embedding[d]
            }
        }

        return (0..<k).map { cluster in
            let count = counts[cluster]
            guard count > 0 else {
                // Empty cluster: reinitialize from random data point (sklearn approach)
                return embeddings.randomElement(using: &rng) ?? sums[cluster]
            }
            var invCount = 1.0 / Double(count)
            var result = [Double](repeating: 0, count: dimension)
            vDSP_vsmulD(sums[cluster], 1, &invCount, &result, 1, vDSP_Length(dimension))
            return result
        }
    }

    /// Seeded random number generator for reproducible clustering.
    /// Uses LCG (Linear Congruential Generator) algorithm.
    struct SeededRNG: RandomNumberGenerator {
        var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }
    }
}
