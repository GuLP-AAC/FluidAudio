import XCTest

@testable import FluidAudio

/// GuLP fork: pyannote's 20% clean-activity rule (`min_active_ratio`) picks the
/// embeddings that train the clustering. Every active speaker still has an
/// embedding, so speakers who only talk in short turns are assigned to a cluster
/// instead of disappearing.
@available(macOS 14.0, iOS 17.0, *)
final class TrainingEmbeddingSelectionTests: XCTestCase {

    private func embedding(ratio: Float, value: Float = 0.5) -> TimedEmbedding {
        TimedEmbedding(
            chunkIndex: 0,
            speakerIndex: 0,
            startFrame: 0,
            endFrame: 10,
            frameWeights: [],
            startTime: 0,
            endTime: 1,
            embedding256: [Float](repeating: value, count: 256),
            rho128: [Double](repeating: 0, count: 128),
            cleanActiveRatio: ratio
        )
    }

    func testSpeakersActiveUnderTwentyPercentDoNotTrain() {
        let manager = OfflineDiarizerManager(config: OfflineDiarizerConfig())
        let selected = manager.selectTrainingEmbeddings(timedEmbeddings: [
            embedding(ratio: 0.6),
            embedding(ratio: 0.1),
            embedding(ratio: 0.2),
            embedding(ratio: 0.05),
        ])
        XCTAssertEqual(selected, [0, 2])
    }

    func testAllFiniteEmbeddingsTrainWhenNoSpeakerIsActiveEnough() {
        let manager = OfflineDiarizerManager(config: OfflineDiarizerConfig())
        let selected = manager.selectTrainingEmbeddings(timedEmbeddings: [
            embedding(ratio: 0.1),
            embedding(ratio: 0.1, value: .nan),
            embedding(ratio: 0.15),
        ])
        XCTAssertEqual(selected, [0, 2])
    }
}
