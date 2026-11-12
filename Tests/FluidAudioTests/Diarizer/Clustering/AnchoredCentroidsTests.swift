import XCTest

@testable import FluidAudio

/// GuLP fork: a known speaker's enrolled embedding (the anchor) pins the centroid
/// that matches it, so that centroid can't drift toward a similar-sounding voice
/// and take its speech.
///
/// The embeddings here are unit vectors in a plane, written as angles: two
/// vectors' cosine is the cosine of the angle between them.
@available(macOS 14.0, iOS 17.0, *)
final class AnchoredCentroidsTests: XCTestCase {

    private func vector(_ degrees: Double) -> [Double] {
        let radians = degrees * .pi / 180
        return [cos(radians), sin(radians), 0]
    }

    private func nearest(_ point: [Double], _ centroids: [[Double]]) -> Int {
        let scores = centroids.map { centroid in zip(point, centroid).reduce(0) { $0 + $1.0 * $1.1 } }
        return scores.indices.max { scores[$0] < scores[$1] }!
    }

    private func assertEqual(_ a: [Double], _ b: [Double], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, file: file, line: line)
        for (x, y) in zip(a, b) { XCTAssertEqual(x, y, accuracy: 1e-9, file: file, line: line) }
    }

    /// The known voice sits at ±2°, the other voice at 44°-90°. A split learned
    /// from the session alone settled with the known voice's centroid at 22°,
    /// holding the other voice's 44°-46° speech: the mean of its members, and
    /// every member nearer it than the other centroid (80°), so re-learning from
    /// the session alone never leaves it. Held at the enrollment, it lets go.
    func testDriftedCentroidGivesBackTheOtherVoicesSpeech() throws {
        let known = [-2.0, 2].map(vector)
        let other = [44.0, 46, 70, 80, 90].map(vector)
        let drifted = [vector(22), vector(80)]
        XCTAssertEqual(nearest(vector(44), drifted), 0, "the scenario needs the drift to hold the 44° speech")
        XCTAssertEqual(nearest(vector(46), drifted), 0, "the scenario needs the drift to hold the 46° speech")

        let refined = try XCTUnwrap(
            AnchoredCentroids.refine(centroids: drifted, trainingEmbeddings: known + other, anchor: vector(0)))

        assertEqual(refined[0], vector(0))
        for point in other { XCTAssertEqual(nearest(point, refined), 1) }
        for point in known { XCTAssertEqual(nearest(point, refined), 0) }
    }

    func testAnchorPinsTheCentroidItMatchesBest() throws {
        let points = [0.0, 10, 80, 90].map(vector)
        let refined = try XCTUnwrap(
            AnchoredCentroids.refine(centroids: [vector(85), vector(5)], trainingEmbeddings: points, anchor: vector(0)))
        assertEqual(refined[1], vector(0))
        assertEqual(refined[0], vector(85))
    }

    /// The known speaker may not be in the recording, or the enrollment may not
    /// match it: nothing changes.
    func testAnchorThatMatchesNoCentroidChangesNothing() {
        let refined = AnchoredCentroids.refine(
            centroids: [vector(70), vector(90)], trainingEmbeddings: [70.0, 80, 90].map(vector), anchor: vector(0))
        XCTAssertNil(refined, "cosine 0.34 is under the \(AnchoredCentroids.minimumSimilarity) minimum")
    }

    /// A voice that loses every embedding keeps its centroid: anchoring never
    /// removes a voice.
    func testCentroidThatLosesEveryEmbeddingKeepsItsPlace() throws {
        let refined = try XCTUnwrap(
            AnchoredCentroids.refine(
                centroids: [vector(10), vector(20)], trainingEmbeddings: [0.0, 2, 4].map(vector), anchor: vector(0)))
        XCTAssertEqual(refined.count, 2)
        assertEqual(refined[0], vector(0))
        assertEqual(refined[1], vector(20))
    }

    /// The anchor's length doesn't matter: embeddings compare by cosine.
    func testAnchorLengthDoesNotMatter() throws {
        let points = [0.0, 3, 50, 60, 70].map(vector)
        let unit = try XCTUnwrap(
            AnchoredCentroids.refine(centroids: [vector(25), vector(70)], trainingEmbeddings: points, anchor: vector(0)))
        let long = try XCTUnwrap(
            AnchoredCentroids.refine(
                centroids: [vector(25), vector(70)], trainingEmbeddings: points, anchor: vector(0).map { $0 * 7 }))
        for (a, b) in zip(unit, long) { assertEqual(a, b) }
    }
}
