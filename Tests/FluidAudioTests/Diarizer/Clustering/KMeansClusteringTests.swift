import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class KMeansClusteringTests: XCTestCase {

    // MARK: - Basic Clustering Tests

    func testKMeansProducesRequestedClusterCount() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ]

        // Use seed for reproducibility (Expert Panel: Crispin)
        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 6)
        let uniqueClusters = Set(clusters)
        XCTAssertEqual(uniqueClusters.count, 3)
    }

    func testKMeansHandlesSingleCluster() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, 0.2],
        ]

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 1,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 3)
        XCTAssertTrue(clusters.allSatisfy { $0 == 0 })
    }

    func testKMeansReturnsSequentialAssignmentsForMoreClustersThanEmbeddings() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 5,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 2)
        XCTAssertEqual(clusters[0], 0)
        XCTAssertEqual(clusters[1], 1)
    }

    func testKMeansComputesCentroids() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]

        let (clusters, centroids) = KMeansClustering.clusterWithCentroids(
            embeddings: embeddings,
            numClusters: 2,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(centroids.count, 2)
        XCTAssertEqual(clusters.count, 4)
    }

    // MARK: - Reproducibility Tests (Expert Panel: Crispin)

    func testKMeansIsDeterministicWithSameSeed() {
        let embeddings: [[Double]] = [
            [1.0, 0.0], [1.1, 0.1], [0.0, 1.0],
            [0.1, 1.1], [-1.0, 0.0], [-0.9, 0.1],
        ]

        let clusters1 = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            seed: 12345
        )

        let clusters2 = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            seed: 12345
        )

        XCTAssertEqual(clusters1, clusters2)
    }

    // MARK: - Realistic Dimension Tests (Expert Panel: Adzic)

    // MARK: - Restarts (GuLP fork)

    /// 30 points around one direction, 3 around another: a single random start
    /// often seeds both centroids in the big group and splits it, leaving the
    /// small one merged in. k-means++ with restarts finds the small group.
    private func imbalancedGroups() -> [[Double]] {
        var embeddings: [[Double]] = []
        for i in 0..<30 {
            let t = Double(i) * 0.01
            embeddings.append([1.0, t, 0.02 * Double(i % 3)])
        }
        for i in 0..<3 {
            embeddings.append([0.0, 1.0, 0.01 * Double(i)])
        }
        return embeddings
    }

    func testRestartsSeparateASmallGroupFromALargeOne() {
        let (clusters, centroids) = KMeansClustering.clusterWithCentroids(
            embeddings: imbalancedGroups(),
            numClusters: 2,
            maxIterations: 100,
            seed: KMeansClustering.defaultSeed,
            restarts: KMeansClustering.defaultRestarts
        )

        XCTAssertEqual(centroids.count, 2)
        let big = Set(clusters[0..<30])
        let small = Set(clusters[30..<33])
        XCTAssertEqual(big.count, 1, "the large group stays together")
        XCTAssertEqual(small.count, 1, "the small group stays together")
        XCTAssertNotEqual(big, small, "the small group gets its own cluster")
    }

    func testRestartsAreDeterministicForTheSameSeed() {
        let embeddings = imbalancedGroups()
        let first = KMeansClustering.clusterWithCentroids(
            embeddings: embeddings, numClusters: 2, maxIterations: 100, seed: 42, restarts: 10)
        let second = KMeansClustering.clusterWithCentroids(
            embeddings: embeddings, numClusters: 2, maxIterations: 100, seed: 42, restarts: 10)
        XCTAssertEqual(first.clusters, second.clusters)
        XCTAssertEqual(first.centroids, second.centroids)
    }

    func testKMeansWithRealisticEmbeddingDimension() {
        let dimension = 192  // Real speaker embedding dimension
        var rng = KMeansClustering.SeededRNG(seed: 42)

        let embeddings = (0..<20).map { _ in
            (0..<dimension).map { _ in Double.random(in: -1...1, using: &rng) }
        }

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 20)
        XCTAssertEqual(Set(clusters).count, 3)
    }
}
