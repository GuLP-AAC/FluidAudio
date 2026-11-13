import XCTest

@testable import FluidAudio

/// A whole recording handed to `processComplete` is complete: its last chunks take whatever
/// right context is left, so the end of the recording gets predictions too.
final class SortformerSessionEndTests: XCTestCase {

    /// Deterministic noise: the loader only cares about how many mel frames there are.
    private func audio(seconds: Double, sampleRate: Int = 16_000) -> [Float] {
        srand48(7)
        return (0..<Int(seconds * Double(sampleRate))).map { _ in Float(drand48() - 0.5) }
    }

    /// The core (new) mel frames each chunk carries, in order.
    private func coreFrames(config: SortformerConfig, audio: [Float]) -> [Int] {
        var loader = SortformerFeatureLoader(config: config, audio: audio)
        var cores: [Int] = []
        while let (features, _, leftOffset, rightOffset) = loader.next() {
            cores.append(features.count / config.melFeatures - leftOffset - rightOffset)
        }
        return cores
    }

    func testBatchLoaderCoversEveryFrameOfTheRecording() {
        for config in [SortformerConfig.fastV2_1, .efficientV2_1, .highContextV2_1] {
            for seconds in [3.1, 5.0, 47.3] {
                let samples = audio(seconds: seconds)
                let (_, melFrames, _) = AudioMelSpectrogram().computeFlatTransposed(audio: samples)
                let cores = coreFrames(config: config, audio: samples)
                XCTAssertEqual(
                    cores.reduce(0, +), melFrames,
                    "chunk \(config.chunkLen), \(seconds) s: the chunks cover \(cores.reduce(0, +)) of \(melFrames) frames")
            }
        }
    }

    func testBatchLoaderCountsTheChunksItReturns() {
        let config = SortformerConfig.efficientV2_1
        let samples = audio(seconds: 5.0)
        let loader = SortformerFeatureLoader(config: config, audio: samples)
        XCTAssertEqual(loader.numChunks, coreFrames(config: config, audio: samples).count)
    }

    func testOnlyTheLastChunksGiveUpRightContext() {
        let config = SortformerConfig.efficientV2_1
        var loader = SortformerFeatureLoader(config: config, audio: audio(seconds: 9.0))
        var rightOffsets: [Int] = []
        while let (_, _, _, rightOffset) = loader.next() { rightOffsets.append(rightOffset) }
        let full = config.chunkRightContext * config.subsamplingFactor
        XCTAssertEqual(rightOffsets.last, 0, "the last chunk has nothing after it")
        XCTAssertEqual(rightOffsets.first, full)
        // Once the right context starts shrinking it never grows back.
        XCTAssertEqual(rightOffsets, rightOffsets.sorted(by: >))
    }

    func testEfficientPresetMatchesTheTwoSecondModel() {
        let config = SortformerConfig.efficientV2_1
        XCTAssertEqual(config.chunkLen, 25)
        XCTAssertEqual(config.chunkLeftContext, 1)
        XCTAssertEqual(config.chunkRightContext, 7)
        XCTAssertEqual(config.fifoLen, 40)
        XCTAssertEqual(config.spkcacheLen, 188)
        XCTAssertEqual(config.spkcacheUpdatePeriod, 31)
        // The model's chunk input: (25 + 1 + 7) frames of 8 mel frames each.
        XCTAssertEqual(config.chunkMelFrames, 264)
        // The file lives under v3/ in the model repository, which the downloader does not know:
        // it is loaded from a local path.
        XCTAssertNil(config.modelVariant)
    }
}
