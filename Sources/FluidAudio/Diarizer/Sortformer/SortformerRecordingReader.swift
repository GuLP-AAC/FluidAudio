@preconcurrency import AVFoundation
import Foundation

/// A complete recording's samples at the model's rate, read forward a block at a time.
protocol SortformerRecordingReader {
    /// The next samples of the recording, in order. Empty at the end of the recording, and on
    /// every call after it.
    mutating func read() throws -> [Float]
}

/// The most samples one read returns: one second at 16 kHz, which is how
/// `AudioConverter.resampleAudioFile` reads a file at that rate.
private let blockSize = 16_000

/// A recording that is already in memory.
struct SortformerRecordingInMemory: SortformerRecordingReader {
    private let samples: [Float]
    private var position = 0

    init(_ samples: [Float]) {
        self.samples = samples
    }

    mutating func read() -> [Float] {
        let end = min(position + blockSize, samples.count)
        defer { position = end }
        return Array(samples[position..<end])
    }
}

/// A recording file that is already mono and at the model's rate, read from disk a second at a
/// time. These are the reads `AudioConverter.resampleAudioFile` makes for such a file, which it
/// does not resample, so the samples are the same ones.
final class SortformerRecordingFile: SortformerRecordingReader {
    /// The length the file reports, for progress. The recording ends where its samples do.
    var expectedSampleCount: Int { Int(file.length) }

    private let file: AVAudioFile
    private let buffer: AVAudioPCMBuffer

    /// Nil when the file has to be converted first: another sample rate, or more than one channel.
    init?(_ file: AVAudioFile, sampleRate: Int) {
        let format = file.processingFormat
        guard format.sampleRate == Double(sampleRate), format.channelCount == 1,
            format.commonFormat == .pcmFormatFloat32, !format.isInterleaved,
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(blockSize))
        else { return nil }
        self.file = file
        self.buffer = buffer
    }

    func read() throws -> [Float] {
        let remaining = file.length - file.framePosition
        guard remaining > 0 else { return [] }
        try file.read(into: buffer, frameCount: AVAudioFrameCount(min(AVAudioFramePosition(blockSize), remaining)))
        guard let samples = buffer.floatChannelData else { throw SortformerError.invalidAudioData }
        return Array(UnsafeBufferPointer(start: samples[0], count: Int(buffer.frameLength)))
    }
}
