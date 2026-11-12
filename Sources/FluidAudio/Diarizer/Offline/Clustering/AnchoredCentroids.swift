import Foundation

/// GuLP fork: pins one speaker's centroid to an embedding enrolled for that
/// speaker (the anchor), then re-learns the other centroids around it.
///
/// Why: centroids learned from a recording alone can drift. When two voices
/// sound alike, the known speaker's centroid absorbs the other voice's most
/// similar speech and moves toward it, which pulls in more of that speech. An
/// enrollment is clean speech from the known speaker only, so a centroid held
/// there keeps the boundary where the voices actually differ.
///
/// All comparisons are cosine similarities, so vector lengths don't matter.
enum AnchoredCentroids {

    /// The least cosine similarity between the anchor and its best-matching
    /// centroid for the anchor to be used. Below it the known speaker is taken
    /// to be absent (or the enrollment not to match), and nothing changes.
    static let minimumSimilarity = 0.5

    /// Stops a re-estimation that keeps moving points back and forth.
    static let maxIterations = 50

    /// The centroids with the one closest to `anchor` replaced by it and the
    /// others re-estimated (the mean of the training embeddings nearest each)
    /// until no embedding changes centroid. A centroid left with no embedding
    /// keeps its last place, so no speaker is removed. Nil when no centroid's
    /// similarity to the anchor reaches `minimumSimilarity`. Returned centroids
    /// are unit length.
    static func refine(
        centroids: [[Double]],
        trainingEmbeddings: [[Double]],
        anchor: [Double]
    ) -> [[Double]]? {
        let anchor = unit(anchor)
        var centroids = centroids.map(unit)
        let similarities = centroids.map { dot($0, anchor) }
        guard let pinned = similarities.indices.max(by: { similarities[$0] < similarities[$1] }),
            similarities[pinned] >= minimumSimilarity
        else { return nil }
        centroids[pinned] = anchor

        let points = trainingEmbeddings.map(unit)
        var assignments: [Int] = []
        for _ in 0..<maxIterations {
            let next = points.map { nearest($0, centroids) }
            if next == assignments { break }
            assignments = next
            for index in centroids.indices where index != pinned {
                let members = zip(points, assignments).filter { $0.1 == index }.map(\.0)
                guard !members.isEmpty else { continue }
                centroids[index] = unit(mean(members))
            }
        }
        return centroids
    }

    private static func nearest(_ point: [Double], _ centroids: [[Double]]) -> Int {
        var best = 0
        var bestScore = -Double.infinity
        for (index, centroid) in centroids.enumerated() {
            let score = dot(point, centroid)
            if score > bestScore {
                bestScore = score
                best = index
            }
        }
        return best
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    private static func mean(_ vectors: [[Double]]) -> [Double] {
        var sum = [Double](repeating: 0, count: vectors[0].count)
        for vector in vectors {
            for i in sum.indices { sum[i] += vector[i] }
        }
        return sum.map { $0 / Double(vectors.count) }
    }

    private static func unit(_ vector: [Double]) -> [Double] {
        let norm = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? vector.map { $0 / norm } : vector
    }
}
