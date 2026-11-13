@preconcurrency import AVFoundation
import XCTest

@preconcurrency @testable import FluidAudio

// Note: Import order is not alphabetical due to Swift 6.1 (CI) vs 6.3 (local) formatter incompatibility.
// OrderedImports rule is disabled in .swift-format until GitHub Actions supports Swift 6.3.

@MainActor
final class SortformerStreamingIntegrationTests: XCTestCase {
    private static var cachedModels: SortformerModels?

    private func loadModelsForTest(config: SortformerConfig) async throws -> SortformerModels {
        if let cachedModels = Self.cachedModels {
            return cachedModels
        }

        let models = try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
        Self.cachedModels = models
        return models
    }

    func testFinalizeSessionMatchesProcessCompleteFrameCount() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let streamingDiarizer = SortformerDiarizer(config: config)
        streamingDiarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try streamingDiarizer.process(samples: chunk)
        }
        let finalChunk = try streamingDiarizer.finalizeSession()

        let completeDiarizer = SortformerDiarizer(config: config)
        completeDiarizer.initialize(models: models)
        let expectedTimeline = try completeDiarizer.processComplete(samples)

        assertBatchReachesTheEnd(
            streaming: streamingDiarizer.timeline.numFinalizedFrames,
            batch: expectedTimeline.numFinalizedFrames,
            config: config, sampleCount: samples.count)
        XCTAssertEqual(finalChunk?.chunkResult.tentativeFrameCount, 0)
        XCTAssertEqual(finalChunk?.chunkResult.tentativePredictions.count, 0)
        XCTAssertEqual(streamingDiarizer.timeline.numTentativeFrames, 0)
    }

    func testFinalizeSessionFlushesTentativeTailAfterAddAudioOnly() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let bufferedDiarizer = SortformerDiarizer(config: config)
        bufferedDiarizer.initialize(models: models)
        bufferedDiarizer.addAudio(samples)
        let bufferedFinalChunk = try bufferedDiarizer.finalizeSession()

        let streamingDiarizer = SortformerDiarizer(config: config)
        streamingDiarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try streamingDiarizer.process(samples: chunk)
        }
        let _ = try streamingDiarizer.finalizeSession()

        XCTAssertNotNil(bufferedFinalChunk)
        XCTAssertEqual(bufferedFinalChunk?.chunkResult.tentativeFrameCount, 0)
        XCTAssertEqual(bufferedDiarizer.timeline.numTentativeFrames, 0)
        XCTAssertEqual(
            bufferedDiarizer.timeline.numFinalizedFrames,
            streamingDiarizer.timeline.numFinalizedFrames
        )
    }

    func testFinalizeSessionIsIdempotent() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)
        for chunk in DiarizationTestFixtures.chunk(samples, sizes: [4_800, 7_680, 9_600]) {
            let _ = try diarizer.process(samples: chunk)
        }

        let firstUpdate = try diarizer.finalizeSession()
        let frameCountAfterFirst = diarizer.timeline.numFinalizedFrames
        let secondUpdate = try diarizer.finalizeSession()

        XCTAssertNotNil(firstUpdate)
        XCTAssertNil(secondUpdate)
        XCTAssertEqual(diarizer.timeline.numFinalizedFrames, frameCountAfterFirst)
    }

    func testFinalizeSessionDrainsUnprocessedAudioAndMelFeatures() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)

        // Feed audio without ever calling process(), so the entire buffer is
        // unprocessed audio at finalize time. finalizeSession must drain it.
        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)
        try diarizer.addAudio(samples, sourceSampleRate: nil)
        let update = try diarizer.finalizeSession()

        let reference = SortformerDiarizer(config: config)
        reference.initialize(models: models)
        let referenceTimeline = try reference.processComplete(samples)

        XCTAssertNotNil(update)
        XCTAssertGreaterThan(diarizer.timeline.numFinalizedFrames, 0)
        XCTAssertEqual(diarizer.timeline.numTentativeFrames, 0)
        assertBatchReachesTheEnd(
            streaming: diarizer.timeline.numFinalizedFrames,
            batch: referenceTimeline.numFinalizedFrames,
            config: config, sampleCount: samples.count)
        // Returned update must cover the freshly-finalized frames produced by
        // the call (drained chunks + absorbed trailing tentative).
        XCTAssertEqual(
            update?.chunkResult.finalizedFrameCount,
            diarizer.timeline.numFinalizedFrames
        )
    }

    func testProcessCompleteStopsWhenItsTaskIsCancelled() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let samples = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 4.0)
        let recording = try self.recording(of: samples, sampleRate: Double(config.sampleRate))
        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)

        let entryPoints: [(String, @Sendable () throws -> DiarizerTimeline)] = [
            ("samples", { try diarizer.processComplete(samples) }),
            ("a file", { try diarizer.processComplete(audioFileURL: recording) }),
        ]
        for (name, processComplete) in entryPoints {
            let stopped = await Task { () -> Bool in
                // Cancelled before the first chunk: nothing may run.
                withUnsafeCurrentTask { $0?.cancel() }
                do {
                    _ = try processComplete()
                    return false
                } catch is CancellationError {
                    return true
                } catch {
                    return false
                }
            }.value

            XCTAssertTrue(stopped, "processComplete from \(name) ran to the end in a canceled task")
        }
    }

    /// A recording file at the model's rate is read from disk a second at a time. Its predictions
    /// are the ones its samples give when they are all handed over at once, bit for bit.
    func testProcessCompleteFromAFileGivesThePredictionsItsSamplesGive() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let fixture = try DiarizationTestFixtures.fixtureAudio(sampleRate: config.sampleRate, limitSeconds: 9.5)
        let recording = try self.recording(of: fixture, sampleRate: Double(config.sampleRate))
        // What the file holds: the fixture as 16-bit samples.
        let samples = try AudioConverter(sampleRate: Double(config.sampleRate)).resampleAudioFile(recording)

        let fromSamples = SortformerDiarizer(config: config)
        fromSamples.initialize(models: models)
        let expected = try fromSamples.processComplete(samples)

        let fromFile = SortformerDiarizer(config: config)
        fromFile.initialize(models: models)
        let timeline = try fromFile.processComplete(audioFileURL: recording)

        XCTAssertGreaterThan(expected.numFinalizedFrames, 0)
        XCTAssertEqual(timeline.numFinalizedFrames, expected.numFinalizedFrames)
        XCTAssertTrue(
            timeline.finalizedPredictions.map(\.bitPattern) == expected.finalizedPredictions.map(\.bitPattern),
            "reading the recording from disk changed the predictions")
    }

    /// A file that is not at the model's rate is still converted in memory, as before.
    func testProcessCompleteFromAFileAtAnotherRateConvertsIt() async throws {
        let config = SortformerConfig.default
        let models: SortformerModels
        do {
            models = try await loadModelsForTest(config: config)
        } catch {
            throw XCTSkip("Sortformer models unavailable in this environment: \(error)")
        }
        let fixture = try DiarizationTestFixtures.fixtureAudio(sampleRate: 44_100, limitSeconds: 4.0)
        let recording = try self.recording(of: fixture, sampleRate: 44_100)
        let samples = try AudioConverter(sampleRate: Double(config.sampleRate)).resampleAudioFile(recording)

        let fromSamples = SortformerDiarizer(config: config)
        fromSamples.initialize(models: models)
        let expected = try fromSamples.processComplete(samples)

        let fromFile = SortformerDiarizer(config: config)
        fromFile.initialize(models: models)
        let timeline = try fromFile.processComplete(audioFileURL: recording)

        XCTAssertGreaterThan(expected.numFinalizedFrames, 0)
        XCTAssertTrue(
            timeline.finalizedPredictions.map(\.bitPattern) == expected.finalizedPredictions.map(\.bitPattern))
    }

    /// Writes samples to a temporary 16-bit PCM CAF file, as a recorder would.
    private func recording(of samples: [Float], sampleRate: Double) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sortformer-recording-\(UUID().uuidString).caf")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: sampleRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsFloatKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(samples.count)))
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let data = try XCTUnwrap(buffer.floatChannelData)[0]
        for (index, sample) in samples.enumerated() { data[index] = sample }
        try file.write(from: buffer)
        return url
    }

    /// The streaming path stops at the last chunk whose right context is whole. A complete
    /// recording (`processComplete`) goes to its last frame, so it is never behind, never more
    /// than one chunk and its right context ahead, and its frames span the whole recording.
    private func assertBatchReachesTheEnd(
        streaming: Int, batch: Int, config: SortformerConfig, sampleCount: Int,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertGreaterThanOrEqual(batch, streaming, file: file, line: line)
        XCTAssertLessThanOrEqual(
            batch - streaming, config.chunkLen + config.chunkRightContext, file: file, line: line)
        let recordingSeconds = Float(sampleCount) / Float(config.sampleRate)
        XCTAssertEqual(
            Float(batch) * config.frameDurationSeconds, recordingSeconds,
            accuracy: config.frameDurationSeconds,
            "processComplete stopped before the end of the recording", file: file, line: line)
    }
}
