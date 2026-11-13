import Foundation

// MARK: - Configuration

/// Configuration for Sortformer streaming diarization.
///
/// Based on NVIDIA's Streaming Sortformer 4-speaker model.
/// Reference: NeMo sortformer_modules.py
public struct SortformerConfig: Sendable {
    public typealias ModelVariant = ModelNames.Sortformer.Variant

    // MARK: - Model Architecture

    public let modelVariant: ModelVariant?

    /// Number of speaker slots (fixed at 4 for current model)
    public let numSpeakers: Int = 4

    /// Pre-encoder embedding dimension
    public let preEncoderDims: Int = 512

    /// Subsampling factor (8:1 downsampling in encoder)
    public let subsamplingFactor: Int = 8

    // MARK: - Streaming Parameters

    /// Output diarization frames per chunk
    /// Must match the value used in CoreML conversion
    public var chunkLen: Int = 6

    /// Left context frames for chunk processing
    public var chunkLeftContext: Int = 1

    /// Right context frames for chunk processing
    public var chunkRightContext: Int = 7

    /// Maximum FIFO queue length (recent embeddings)
    /// Must match CoreML conversion: fifo_len=40
    public var fifoLen: Int = 40

    /// Maximum speaker cache length (historical embeddings)
    /// Must match CoreML conversion: spkcache_len=188
    public var spkcacheLen: Int = 188

    /// Period for speaker cache updates (frames)
    public var spkcacheUpdatePeriod: Int = 31

    /// Silence frames per speaker in compressed cache
    public var spkcacheSilFramesPerSpk: Int = 3

    // MARK: - Debug

    /// Enable debug logging
    public var debugMode: Bool = false

    // MARK: - Audio Parameters

    /// Sample rate in Hz
    public let sampleRate: Int = 16000

    /// Mel spectrogram window size in samples (25ms)
    public let melWindow: Int = 400

    /// Mel spectrogram stride in samples (10ms)
    public let melStride: Int = 160

    /// Number of mel filterbank features
    public let melFeatures: Int = 128

    // MARK: - Thresholds

    /// Threshold for silence detection (sum of speaker probs)
    public var silenceThreshold: Float = 0.2

    /// Threshold for speech prediction
    public var predScoreThreshold: Float = 0.25

    /// Boost factor for latest frames in cache compression
    public var scoresBoostLatest: Float = 0.05

    /// Strong boost rate for top-k selection
    public var strongBoostRate: Float = 0.75

    /// Weak boost rate for preventing speaker dominance
    public var weakBoostRate: Float = 1.5

    /// Minimum positive scores rate
    public var minPosScoresRate: Float = 0.5

    /// Maximum index placeholder for disabled frames in spkcache compression
    public let maxIndex: Int = 99999

    // MARK: - Computed Properties

    /// Total chunk frames for CoreML model input (includes left/right context)
    /// Formula: (chunk_len + left_context + right_context) * subsampling
    /// Default: (6 + 1 + 1) * 8 = 64 frames
    public var chunkMelFrames: Int {
        (chunkLen + chunkLeftContext + chunkRightContext) * subsamplingFactor
    }

    /// Core frames per chunk (without context)
    public var coreFrames: Int {
        chunkLen * subsamplingFactor
    }

    /// Frame duration in seconds
    public var frameDurationSeconds: Float {
        Float(subsamplingFactor) * Float(melStride) / Float(sampleRate)
    }

    // MARK: - Initialization

    /// Configuration matching Gradient Descent's Streaming-Sortformer-Conversion models
    public static let `default` = SortformerConfig(
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2 weights (~1.04s latency, smallest context).
    /// May handle high-speaker-count scenarios better than v2.1 (v2.1 can degrade when many speakers overlap).
    public static let `fastV2` = SortformerConfig(
        modelVariant: .fastV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2.1 weights (~1.04s latency, smallest context).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let `fastV2_1` = SortformerConfig(
        modelVariant: .fastV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Balanced config with Sortformer v2 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM. May handle high-speaker-count scenarios better than v2.1.
    public static let balancedV2 = SortformerConfig(
        modelVariant: .balancedV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// Balanced config with Sortformer v2.1 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM.
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let balancedV2_1 = SortformerConfig(
        modelVariant: .balancedV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// High-context config with Sortformer v2 weights (~30.4s latency, most context window).
    /// May handle high-speaker-count scenarios better than v2.1.
    public static let highContextV2 = SortformerConfig(
        modelVariant: .highContextV2,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// High-context config with Sortformer v2.1 weights (~30.4s latency, most context window).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let highContextV2_1 = SortformerConfig(
        modelVariant: .highContextV2_1,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// Two-second chunks with Sortformer v2.1 weights (upstream FluidAudio's `efficientV2_1`): the
    /// same context as `fastV2_1`, with each model call advancing 2 s of audio instead of 0.48 s.
    /// The cost of a call is mostly the speaker cache and FIFO, which don't grow, so this runs
    /// about four times faster than `fastV2_1` for a recording that is already complete.
    ///
    /// No `modelVariant`: its model, `SortformerEfficient_v2.1.mlmodelc`, sits under `v3/fp16/`
    /// in the model repository, where this fork's downloader doesn't look. Load it from a local
    /// path (`SortformerModels(config:main:)`).
    public static let efficientV2_1 = SortformerConfig(
        modelVariant: nil,
        chunkLen: 25,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// - Warning: If you don't use one of the default configurations, you must use a local model converted with that configuration.
    public init(
        modelVariant: ModelVariant? = .fastV2_1,
        chunkLen: Int = 6,
        chunkLeftContext: Int = 1,
        chunkRightContext: Int = 7,
        fifoLen: Int = 40,
        spkcacheLen: Int = 188,
        spkcacheUpdatePeriod: Int = 31,
        silenceThreshold: Float = 0.2,
        spkcacheSilFramesPerSpk: Int = 3,
        predScoreThreshold: Float = 0.25,
        scoresBoostLatest: Float = 0.05,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        debugMode: Bool = false
    ) {
        self.modelVariant = modelVariant
        self.chunkLen = max(1, chunkLen)
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.fifoLen = fifoLen
        self.silenceThreshold = silenceThreshold
        self.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        self.debugMode = debugMode
        self.predScoreThreshold = predScoreThreshold
        self.scoresBoostLatest = scoresBoostLatest
        self.strongBoostRate = strongBoostRate
        self.weakBoostRate = weakBoostRate
        self.minPosScoresRate = minPosScoresRate

        // The following parameters must meet certain constraints
        self.spkcacheLen = max(spkcacheLen, (1 + self.spkcacheSilFramesPerSpk) * self.numSpeakers)
        self.spkcacheUpdatePeriod = max(min(spkcacheUpdatePeriod, self.fifoLen + self.chunkLen), self.chunkLen)
    }

    public func isCompatible(with other: SortformerConfig) -> Bool {
        return
            (self.chunkMelFrames == other.chunkMelFrames && self.fifoLen == other.fifoLen
            && self.spkcacheLen == other.spkcacheLen)
    }
}

// MARK: - Streaming State

/// State maintained across streaming chunks for Sortformer diarization.
///
/// This mirrors NeMo's StreamingSortformerState dataclass.
/// Reference: NeMo sortformer_modules.py
public struct SortformerStreamingState: Sendable {
    /// Speaker cache embeddings from start of audio
    /// Shape: [spkcacheLen, fcDModel] (e.g., [188, 512])
    public var spkcache: [Float]

    /// Current valid length of speaker cache
    public var spkcacheLength: Int

    /// Speaker predictions for cached embeddings
    /// Shape: [spkcacheLen, numSpeakers] (e.g., [188, 4])
    public var spkcachePreds: [Float]?

    /// FIFO queue of recent chunk embeddings
    /// Shape: [fifoLen, fcDModel] (e.g., [188, 512])
    public var fifo: [Float]

    /// Current valid length of FIFO queue
    public var fifoLength: Int

    /// Speaker predictions for FIFO embeddings
    /// Shape: [fifoLen, numSpeakers] (e.g., [188, 4])
    public var fifoPreds: [Float]?

    /// Running mean of silence embeddings
    /// Shape: [fcDModel] (e.g., [512])
    public var meanSilenceEmbedding: [Float]

    /// Count of silence frames observed
    public var silenceFrameCount: Int

    /// Initialize empty streaming state
    public init(config: SortformerConfig) {
        self.spkcache = []
        self.spkcachePreds = nil
        self.spkcacheLength = 0

        self.fifo = []
        self.fifoPreds = nil
        self.fifoLength = 0

        self.fifo.reserveCapacity((config.fifoLen + config.chunkLen) * config.preEncoderDims)
        self.spkcache.reserveCapacity((config.spkcacheLen + config.spkcacheUpdatePeriod) * config.preEncoderDims)

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: config.preEncoderDims)
        self.silenceFrameCount = 0
    }

    public mutating func cleanup() {
        self.fifo.removeAll(keepingCapacity: false)
        self.spkcache.removeAll(keepingCapacity: false)
        self.fifoPreds = nil
        self.spkcachePreds = nil
        self.spkcacheLength = 0
        self.fifoLength = 0
        self.meanSilenceEmbedding.removeAll(keepingCapacity: false)
        self.silenceFrameCount = 0
    }
}

// MARK: - Feature Loader for a Complete Recording

/// Cuts a complete recording into the model's chunks of mel features.
///
/// The recording is read forward and each chunk's features are computed from only the samples
/// that chunk needs, so what is held does not grow with the recording's length: about three
/// seconds of audio for the two-second chunk model. A mel frame depends only on the window of
/// samples around it, so the chunks are, bit for bit, the ones the spectrogram of the whole
/// recording would be cut into.
public struct SortformerFeatureLoader {
    private let lc: Int
    private let rc: Int
    private let chunkLen: Int
    private let hop: Int
    private let window: Int

    private let melSpectrogram = AudioMelSpectrogram()
    private var reader: any SortformerRecordingReader

    /// The samples still needed, starting at the recording's sample `heldFrom`.
    private var held: [Float] = []
    private var heldFrom = 0
    /// The recording's length, once the reader has run dry.
    private var sampleCount: Int?

    private var startFeat = 0

    public init(config: SortformerConfig, audio: [Float]) {
        self.init(config: config, reader: SortformerRecordingInMemory(audio))
    }

    init(config: SortformerConfig, reader: any SortformerRecordingReader) {
        self.lc = config.chunkLeftContext * config.subsamplingFactor
        self.rc = config.chunkRightContext * config.subsamplingFactor
        self.chunkLen = config.chunkLen * config.subsamplingFactor
        self.hop = config.melStride
        self.window = config.melWindow
        self.reader = reader
    }

    public mutating func next() throws -> (
        chunkFeatures: [Float], chunkLength: Int, leftOffset: Int, rightOffset: Int
    )? {
        // The spectrogram pads the recording by half an FFT on each side, so frame `f` reads the
        // samples from `f * hop - pad` on.
        let pad = melSpectrogram.nFFT / 2

        // Read as far as this chunk and a full right context reach, or to the recording's end.
        let wanted = (startFeat + chunkLen + rc - 1) * hop + pad
        while sampleCount == nil, heldFrom + held.count < wanted {
            let block = try reader.read()
            if block.isEmpty {
                sampleCount = heldFrom + held.count
            } else {
                held.append(contentsOf: block)
            }
        }

        // The recording's frames, as the spectrogram of the whole of it counts them. Until the end
        // is known there are more than this chunk can use.
        let featLength: Int
        if let sampleCount {
            featLength = sampleCount == 0 ? 0 : 1 + (sampleCount + 2 * pad - window) / hop
        } else {
            featLength = .max
        }

        // Calculate end of core chunk
        let endFeat = min(startFeat + chunkLen, featLength)

        // Need at least one core frame
        guard endFeat > startFeat else { return nil }

        // The audio is complete, so the last chunks take whatever right context is left, down to
        // none. Waiting for a full right context (as the streaming path must) dropped the end of
        // the recording: up to a chunk and its right context, half a minute for the high-context
        // model.
        let leftOffset = min(lc, startFeat)
        let rightOffset = min(rc, featLength - endFeat)

        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        let chunkLength = chunkEndFrame - chunkStartFrame

        // The padded samples these frames read: zeros before the recording's first sample, and
        // nothing after its last, which the spectrogram reads as zeros.
        let from = chunkStartFrame * hop - pad
        let to = min(heldFrom + held.count, (chunkEndFrame - 1) * hop + pad)
        let firstSample = max(from, 0)
        var samples = [Float](repeating: 0, count: firstSample - from)
        samples.append(contentsOf: held[(firstSample - heldFrom)..<(to - heldFrom)])
        // Preemphasis reaches one sample back.
        let sampleBefore = firstSample > 0 ? held[firstSample - 1 - heldFrom] : 0
        let (chunkFeatures, _, _) = melSpectrogram.computeFlatTransposed(
            audio: samples, lastAudioSample: sampleBefore, paddingMode: .prePadded, expectedFrameCount: chunkLength)

        startFeat = endFeat

        // Let go of the samples before the next chunk's left context.
        let nextFrom = max(0, (startFeat - min(lc, startFeat)) * hop - pad - 1)
        if nextFrom > heldFrom {
            held.removeFirst(min(held.count, nextFrom - heldFrom))
            heldFrom = nextFrom
        }

        return (chunkFeatures, chunkLength, leftOffset, rightOffset)
    }
}

// MARK: - Result Types

/// Result from streaming state update containing both confirmed and tentative predictions.
///
/// - `confirmed`: Predictions for frames that have passed beyond the right context window.
///   These are final and will not change.
/// - `tentative`: Predictions for frames still within the right context window.
///   These may change when the next chunk arrives with more future context.
///
/// This enables real-time UI display without waiting for the full right context delay.
/// With rightContext=7 and 80ms frames, tentative predictions provide 560ms earlier feedback.
public struct StreamingUpdateResult: Sendable {
    /// Final predictions for confirmed frames [chunkLen * numSpeakers]
    public let confirmed: [Float]

    /// Tentative predictions for right context frames [rightContext * numSpeakers]
    /// May change with next chunk. Empty if rightContext=0.
    public let tentative: [Float]

    /// Number of speakers
    public let numSpeakers: Int

    /// Number of confirmed frames
    public var confirmedFrameCount: Int { confirmed.count / numSpeakers }  // Assumes 4 speakers

    /// Number of tentative frames
    public var tentativeFrameCount: Int { tentative.count / numSpeakers }  // Assumes 4 speakers

    public init(confirmed: [Float], tentative: [Float], numSpeakers: Int = 4) {
        self.confirmed = confirmed
        self.tentative = tentative
        self.numSpeakers = numSpeakers
    }
}

// MARK: - Errors

public enum SortformerError: Error, LocalizedError {
    case notInitialized
    case modelLoadFailed(String)
    case preprocessorFailed(String)
    case inferenceFailed(String)
    case invalidAudioData
    case invalidState(String)
    case configurationError(String)
    case insufficientChunkLength(String)
    case insufficientPredsLength(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Sortformer diarizer not initialized. Call initialize() first."
        case .modelLoadFailed(let message):
            return "Failed to load Sortformer model: \(message)"
        case .preprocessorFailed(let message):
            return "Preprocessor failed: \(message)"
        case .inferenceFailed(let message):
            return "Inference failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .insufficientChunkLength(let message):
            return "Insufficient chunk length: \(message)"
        case .insufficientPredsLength(let message):
            return "Insufficient preds length: \(message)"
        }
    }
}
