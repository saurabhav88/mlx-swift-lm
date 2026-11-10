import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Compile probe for what a speculative decoder in another module reads from the Qwen 3.5
/// text model.
///
/// This file imports `MLXLLM` without `@testable` on purpose, so it sees only the public
/// surface. It stops compiling if the embedding, the output head, the layer-kind flag or the
/// capturing forward is no longer public.
enum Qwen35DFlashPublicAPIProbe {
    static func embed(_ model: Qwen35TextModel, _ tokens: MLXArray) -> MLXArray {
        model.model.embedTokens(tokens)
    }

    static func logits(_ model: Qwen35TextModel, _ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead = model.lmHead {
            return lmHead(hiddenStates)
        }
        return model.model.embedTokens.asLinear(hiddenStates)
    }

    static func forwardWithCapture(
        _ model: Qwen35TextModel,
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.model.callCapturing(
            inputIDs, cache: cache.map { $0 }, captureLayerIDs: captureLayerIDs)
        return (logits(model, hiddenStates), captured)
    }

    static func isHybrid(_ model: Qwen35TextModel) -> Bool {
        model.model.hasLinearLayers
    }
}
