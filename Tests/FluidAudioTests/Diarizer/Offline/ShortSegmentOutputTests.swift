import XCTest

@testable import FluidAudio

/// The shortest segment the diarizer outputs is its own setting, apart from how
/// much speech a voiceprint needs. One shared value dropped every turn under a
/// second ("more", "go") from the output.
@available(macOS 14.0, iOS 17.0, *)
final class ShortSegmentOutputTests: XCTestCase {

    private let frameDuration = 0.1

    /// One 6 s window: voice 0 talks for 2.0 s, voice 1 answers for 0.6 s,
    /// voice 1 makes a 0.2 s sound, then voice 0 talks for 1.5 s.
    private func segmentation() -> SegmentationOutput {
        var weights = [[Float]](repeating: [0, 0], count: 60)
        for frame in 0..<20 { weights[frame] = [1, 0] }
        for frame in 26..<32 { weights[frame] = [0, 1] }
        for frame in 38..<40 { weights[frame] = [0, 1] }
        for frame in 45..<60 { weights[frame] = [1, 0] }
        return SegmentationOutput(
            logProbs: [],
            speakerWeights: [weights],
            numChunks: 1,
            numFrames: 60,
            numSpeakers: 2,
            chunkOffsets: [0],
            frameDuration: frameDuration
        )
    }

    private func durations(_ config: OfflineDiarizerConfig) -> [Double] {
        OfflineReconstruction(config: config)
            .buildSegments(
                segmentation: segmentation(),
                hardClusters: [[0, 1]],
                centroids: [[1, 0], [0, 1]]
            )
            .sorted { $0.startTimeSeconds < $1.startTimeSeconds }
            .map { (Double($0.durationSeconds) * 10).rounded() / 10 }
    }

    func testAnswerUnderOneSecondIsKeptWhenTheOutputMinimumAllowsIt() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = 0.3

        XCTAssertEqual(durations(config), [2.0, 0.6, 1.5])
    }

    func testSoundShorterThanTheOutputMinimumIsStillDropped() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = 0.1

        XCTAssertEqual(durations(config), [2.0, 0.6, 0.2, 1.5])

        config.postProcessing.minSegmentDurationSeconds = 0.3
        XCTAssertFalse(durations(config).contains(0.2))
    }

    func testWithNoOutputMinimumTheVoiceprintMinimumDecidesAsBefore() {
        let config = OfflineDiarizerConfig()

        XCTAssertNil(config.postProcessing.minSegmentDurationSeconds)
        XCTAssertEqual(durations(config), [2.0, 1.5])
    }

    func testOutputMinimumLeavesTheVoiceprintMinimumAlone() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = 0.3

        XCTAssertEqual(config.embedding.minSegmentDurationSeconds, 1.0)
        XCTAssertEqual(config.minOutputSegmentDuration, 0.3)
    }

    /// The speaker-count floor and the anchor ask how many voices the session has.
    /// A voice that only made a sub-second sound must not count, or a session that
    /// collapsed to one voice passes the floor and is never re-clustered.
    func testVoiceWithOnlyAShortSoundIsNotCountedAsAVoice() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = 0.1
        let segments = OfflineReconstruction(config: config).buildSegments(
            segmentation: segmentation(),
            hardClusters: [[0, 1]],
            centroids: [[1, 0], [0, 1]]
        )

        XCTAssertEqual(Set(segments.map(\.speakerId)).count, 2)
        XCTAssertEqual(OfflineReconstruction(config: config).establishedVoiceCount(in: segments), 1)
    }

    func testVoiceWithATurnLongEnoughForAVoiceprintIsCounted() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = 0.1
        config.embedding.minSegmentDurationSeconds = 0.5
        let segments = OfflineReconstruction(config: config).buildSegments(
            segmentation: segmentation(),
            hardClusters: [[0, 1]],
            centroids: [[1, 0], [0, 1]]
        )

        XCTAssertEqual(OfflineReconstruction(config: config).establishedVoiceCount(in: segments), 2)
    }

    func testNegativeOutputMinimumIsRejected() {
        var config = OfflineDiarizerConfig()
        config.postProcessing.minSegmentDurationSeconds = -0.1

        XCTAssertThrowsError(try config.validate())
    }
}
