@preconcurrency import AVFoundation
import XCTest

@testable import FluidAudio

/// A complete recording is read forward and its mel features are computed a chunk at a time, so
/// the memory a recording needs does not grow with its length. The chunks are the ones the
/// whole-recording spectrogram would have been cut into, bit for bit.
final class SortformerBoundedLoaderTests: XCTestCase {

    /// Deterministic noise.
    private func audio(count: Int) -> [Float] {
        srand48(11)
        return (0..<count).map { _ in Float(drand48() - 0.5) }
    }

    /// Hands a recording out in blocks, and counts what it has handed out.
    private final class CountingReader: SortformerRecordingReader {
        private let samples: [Float]
        private let block: Int
        private(set) var handedOut = 0

        init(_ samples: [Float], block: Int) {
            self.samples = samples
            self.block = block
        }

        func read() -> [Float] {
            let end = min(handedOut + block, samples.count)
            defer { handedOut = end }
            return Array(samples[handedOut..<end])
        }
    }

    private struct Chunk: Equatable {
        let features: [UInt32]
        let length: Int
        let leftOffset: Int
        let rightOffset: Int
    }

    /// The chunks of a recording, cut from the spectrogram of the whole recording: each chunk is
    /// `chunkLen` new frames with up to a full left and right context around them.
    private func chunksOfTheWholeSpectrogram(config: SortformerConfig, samples: [Float]) -> [Chunk] {
        guard !samples.isEmpty else { return [] }
        let (mel, frames, _) = AudioMelSpectrogram().computeFlatTransposed(audio: samples)
        let core = config.chunkLen * config.subsamplingFactor
        let left = config.chunkLeftContext * config.subsamplingFactor
        let right = config.chunkRightContext * config.subsamplingFactor
        var chunks: [Chunk] = []
        for start in stride(from: 0, to: frames, by: core) {
            let end = min(start + core, frames)
            let leftOffset = min(left, start)
            let rightOffset = min(right, frames - end)
            let slice = mel[((start - leftOffset) * config.melFeatures)..<((end + rightOffset) * config.melFeatures)]
            chunks.append(
                Chunk(
                    features: slice.map(\.bitPattern), length: end + rightOffset - start + leftOffset,
                    leftOffset: leftOffset, rightOffset: rightOffset))
        }
        return chunks
    }

    private func chunks(of loader: inout SortformerFeatureLoader) throws -> [Chunk] {
        var chunks: [Chunk] = []
        while let (features, length, leftOffset, rightOffset) = try loader.next() {
            chunks.append(
                Chunk(
                    features: features.map(\.bitPattern), length: length,
                    leftOffset: leftOffset, rightOffset: rightOffset))
        }
        return chunks
    }

    func testEveryChunkEqualsItsSliceOfTheWholeRecordingsSpectrogram() throws {
        // Around a hop (160), a window (400), a two-second chunk (32,000) and its right context,
        // and lengths that are none of those.
        let counts = [
            1, 159, 160, 161, 399, 400, 401, 31_999, 32_000, 32_001, 40_959, 40_960, 40_961, 80_000, 756_800,
        ]
        for config in [SortformerConfig.fastV2_1, .efficientV2_1, .highContextV2_1] {
            for count in counts {
                let samples = audio(count: count)
                let expected = chunksOfTheWholeSpectrogram(config: config, samples: samples)
                for block in [997, 16_000, count] {
                    let reader = CountingReader(samples, block: block)
                    var loader = SortformerFeatureLoader(config: config, reader: reader)
                    let got = try chunks(of: &loader)
                    XCTAssertEqual(
                        got.count, expected.count, "chunk \(config.chunkLen), \(count) samples, read \(block)")
                    XCTAssertTrue(
                        got == expected,
                        "chunk \(config.chunkLen), \(count) samples, read \(block) at a time: the chunks differ")
                }
            }
        }
    }

    func testSamplesAlreadyInMemoryGiveTheSameChunks() throws {
        let config = SortformerConfig.efficientV2_1
        let samples = audio(count: 100_000)
        var loader = SortformerFeatureLoader(config: config, audio: samples)
        XCTAssertTrue(try chunks(of: &loader) == chunksOfTheWholeSpectrogram(config: config, samples: samples))
    }

    func testAnEmptyRecordingHasNoChunks() throws {
        var loader = SortformerFeatureLoader(config: .efficientV2_1, reader: CountingReader([], block: 16_000))
        XCTAssertNil(try loader.next())
    }

    /// What bounds the memory: when a chunk is returned, the recording has been read no further
    /// than that chunk, its right context and one block ahead.
    func testTheLoaderReadsNoFurtherThanTheChunkItReturnsNeeds() throws {
        let config = SortformerConfig.efficientV2_1
        let block = 16_000
        let samples = audio(count: 60 * config.sampleRate)
        let reader = CountingReader(samples, block: block)
        var loader = SortformerFeatureLoader(config: config, reader: reader)

        let coreSamples = config.chunkLen * config.subsamplingFactor * config.melStride
        let rightContextSamples = config.chunkRightContext * config.subsamplingFactor * config.melStride
        var returned = 0
        while try loader.next() != nil {
            returned += 1
            let needed = returned * coreSamples + rightContextSamples + AudioMelSpectrogram().nFFT
            XCTAssertLessThanOrEqual(
                reader.handedOut, min(samples.count, needed + block),
                "after chunk \(returned) the loader had read \(reader.handedOut) samples")
        }
        XCTAssertEqual(reader.handedOut, samples.count, "the whole recording is read by the end")
        XCTAssertEqual(returned, chunksOfTheWholeSpectrogram(config: config, samples: samples).count)
    }

    // MARK: - Reading a recording file

    /// Writes 16-bit PCM noise to a temporary CAF file, as a recorder would.
    private func recording(seconds: Double, sampleRate: Double, channels: AVAudioChannelCount = 1) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sortformer-reader-\(UUID().uuidString).caf")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: Int(channels), AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsFloatKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let frames = AVAudioFrameCount(seconds * sampleRate)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frames))
        buffer.frameLength = frames
        srand48(5)
        for channel in 0..<Int(channels) {
            let data = try XCTUnwrap(buffer.floatChannelData)[channel]
            for frame in 0..<Int(frames) { data[frame] = Float(drand48() - 0.5) }
        }
        try file.write(from: buffer)
        return url
    }

    func testAFileAtTheModelsRateIsReadForwardAsTheSamplesTheConverterGives() throws {
        // 3.3 s: three whole one-second reads and a short one.
        let url = try recording(seconds: 3.3, sampleRate: 16_000)
        let expected = try AudioConverter(sampleRate: 16_000).resampleAudioFile(url)

        let reader = try XCTUnwrap(SortformerRecordingFile(try AVAudioFile(forReading: url), sampleRate: 16_000))
        XCTAssertEqual(reader.expectedSampleCount, expected.count)
        var samples: [Float] = []
        var reads = 0
        while true {
            let block = try reader.read()
            if block.isEmpty { break }
            samples.append(contentsOf: block)
            reads += 1
        }
        XCTAssertEqual(reads, 4, "a second at a time")
        XCTAssertTrue(samples.map(\.bitPattern) == expected.map(\.bitPattern))
        XCTAssertTrue(try reader.read().isEmpty, "the end stays the end")
    }

    func testAFileThatNeedsConvertingIsNotReadDirectly() throws {
        let otherRate = try AVAudioFile(forReading: try recording(seconds: 0.5, sampleRate: 44_100))
        XCTAssertNil(SortformerRecordingFile(otherRate, sampleRate: 16_000))
        let stereo = try AVAudioFile(forReading: try recording(seconds: 0.5, sampleRate: 16_000, channels: 2))
        XCTAssertNil(SortformerRecordingFile(stereo, sampleRate: 16_000))
    }
}
