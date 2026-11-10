import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

@Test
func testQwen35TextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35VLMTextConfigurationDecodesMTPFields() throws {
    let cfg = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))

    #expect(cfg.mtpNumHiddenLayers == 1)
    #expect(cfg.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let sanitized = drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 32, 16]),
        "mtp.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 16, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.gate_up_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.down_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
    let norm = try #require(sanitized["mtp.norm.weight"])
    let pre = try #require(sanitized["mtp.pre_fc_norm_embedding.weight"])
    eval(norm, pre)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(pre, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(qwen35StandaloneMTPConfigJSON().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = drafter.sanitize(weights: ["mtp.norm.weight": weight])
    let norm = try #require(sanitized["mtp.norm.weight"])
    eval(norm)
    #expect(allClose(norm, weight, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35MTPDraftSanitizeStacksPerExpertMoEWeights() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(qwen35TextConfigJSON(mtpLayers: 1, numExperts: 2).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    let weights: [String: MLXArray] = [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": MLXArray.ones([16, 16]),
    ]

    let sanitized = drafter.sanitize(weights: weights)

    #expect(sanitized["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
}

@Test
func testQwen35MTPDraftInstantiatesDedicatedEmbeddingWhenConfigured() throws {
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            qwen35TextConfigJSON(mtpLayers: 1, mtpUseDedicatedEmbeddings: true).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)

    #expect(drafter.mtp.embedTokens != nil)
    let sanitized = drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

/// Bound for comparing a gated-delta-net state reached two ways. Wider on machines whose
/// float32 matmuls run as TF32; see ``MatmulPrecision``.
private let gdnStateTolerance = MatmulPrecision.tolerance(float32: 1e-5, reduced: 2e-3)

@Suite(.serialized)
struct Qwen35MTPMetalTests {
    @Test
    func testQwen35MTPPredictorAdvancesEveryLayerCachePerToken() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 2).utf8))
        let predictor = MLXLLM.Qwen35MTPPredictor(cfg)
        let cache = predictor.newCache()
        let embeds = MLXArray.zeros([1, 1, 16])
        let hidden = MLXArray.zeros([1, 1, 16])

        let first = predictor(
            inputsEmbeds: embeds, hiddenStates: hidden, cache: cache,
            positionOffset: 128)
        eval(first)
        #expect(cache[0].offset == 1)
        #expect(cache[1].offset == 1)

        let second = predictor(
            inputsEmbeds: embeds, hiddenStates: first, cache: cache,
            positionOffset: 129)
        eval(second)
        #expect(cache[0].offset == 2)
        #expect(cache[1].offset == 2)
    }

    @Test
    func testQwen35VLMMTPPositionIdsApplyMultimodalDelta() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 2,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 2, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[1, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[2, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[1, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[2, 1, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35VLMMTPPositionIdsRepeatAndTrimShortBatchDeltas() throws {
        let positionIds = MLXVLM.qwen35MTPPositionIds(
            offset: 10,
            batchSize: 4,
            positionDeltas: MLXArray([Int32(3), 5])
        )

        eval(positionIds)
        #expect(positionIds.shape == [3, 4, 1])
        #expect(positionIds[0, 0, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 1, 0].item(Int32.self) == 15)
        #expect(positionIds[0, 2, 0].item(Int32.self) == 13)
        #expect(positionIds[0, 3, 0].item(Int32.self) == 15)
    }

    @Test
    func testQwen35TextModelEmitDrafterStateBySynthetic() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let cache = try model.newCache(parameters: nil as GenerateParameters?)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let input = LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        let out = model(input, cache: cache, state: state)

        let hidden = try #require(out.state?[mtpLastHiddenStatesKey])
        let sharedKV = try #require(out.state?[mtpSharedKVStatesKey])
        let sharedKVOffsets = try #require(out.state?[mtpSharedKVOffsetsKey])
        eval(out.logits, hidden)
        #expect(out.logits.shape == [1, 4, 16])
        #expect(hidden.shape == [1, 4, 16])
        #expect(Set(sharedKV.keys) == ["full_attention"])
        #expect(sharedKVOffsets == ["full_attention": 4])
        let full = try #require(sharedKV["full_attention"])
        eval(full.0, full.1)
        #expect(full.0.shape.count == 4)
        #expect(full.1.shape.count == 4)
    }

    @Test
    func testQwen35TextModelEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let model = MLXLLM.Qwen35TextModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let expected = model.model.forward(tokens, applyFinalNorm: false)
        let normalized = model.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: state)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35VLMEmitsPostFinalNormHiddenState() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8))
        let model = MLXVLM.Qwen35(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])
        let base = MLXArray(0 ..< 4).asType(.int32).reshaped([1, 1, 4])
        let positionIds = broadcast(base, to: [3, 1, 4])
        let expected = model.languageModel.model(
            tokens, positionIds: positionIds, applyFinalNorm: false)
        let normalized = model.languageModel.model.norm(expected)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true

        let output = model.languageModel(
            tokens, cache: nil, state: state, positionIds: positionIds)
        let emitted = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(expected, normalized, emitted)

        #expect(allClose(emitted, normalized, rtol: 0, atol: 0).item(Bool.self))
        #expect(!allClose(emitted, expected, rtol: 0, atol: 0).item(Bool.self))
    }

    @Test
    func testQwen35DrafterCacheTracksVerifiedSequenceAcrossAcceptRejectPatterns() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let target = MLXLLM.Qwen35TextModel(cfg)
        let drafter = MLXLLM.Qwen35MTPDraftModel(cfg)
        let sampler = GenerateParameters(temperature: 0).sampler()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])

        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let targetOutput = target(LMInput.Text(tokens: prompt), cache: nil, state: targetState)
        let promptHidden = try #require(targetOutput.state?[mtpLastHiddenStatesKey])

        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            var state = drafter.makeState(parameters: nil)
            var bonus = MLXArray([Int32(4)])
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt, targetHidden: promptHidden,
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: sampler)
            eval(state.seedToken!, state.seedHidden!)
            #expect(state.cache.allSatisfy { $0.offset == 3 })
            #expect(state.nextPosition == 3)

            var expectedPosition = 3
            for accepted in pattern {
                let proposal = drafter.draftBlock(
                    target: target, lastToken: bonus,
                    lastHidden: promptHidden[0..., (-1)..., 0...], sharedKV: [:],
                    positionDeltas: nil, queryOffset: expectedPosition, blockSize: 2,
                    state: &state, sampler: sampler)
                eval(proposal)
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })

                let verifyHidden = MLXArray.zeros([1, 2, cfg.hiddenSize])
                let finalToken = MLXArray([Int32(8 + accepted)])
                drafter.commitDrafterState(
                    target: target, targetHidden: verifyHidden, draftTokens: proposal,
                    acceptedCount: accepted, finalToken: finalToken, positionDeltas: nil,
                    state: &state, sampler: sampler)
                eval(state.seedToken!, state.seedHidden!)
                expectedPosition += accepted + 1
                #expect(state.cache.allSatisfy { $0.offset == expectedPosition })
                #expect(state.nextPosition == expectedPosition)
                bonus = finalToken
            }
        }
    }

    @Test
    func testQwen35GDNCheckpointMatchesPrefixWithoutReplayingProjections() throws {
        let cfg = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXLLM.Qwen35GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.hasSpeculativeCheckpoint)
        #expect(speculativeCache.restoreSpeculativeCheckpoint())

        let restored = speculativeCache.state
        let expected = prefixCache.state
        #expect(restored.count == expected.count)
        for (actual, reference) in zip(restored, expected) {
            eval(actual, reference)
            // The checkpoint and the prefix run reach the same state through
            // differently shaped matmuls, so the bound follows the machine's
            // float32 matmul precision; see MatmulPrecision.
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35VLMGDNCheckpointMatchesPrefix() throws {
        let cfg = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8))
        let layer = MLXVLM.Qwen35Language.GatedDeltaNet(cfg)
        let input = MLXRandom.normal([1, 2, 16])
        let fullCache = MambaCache()
        let speculativeCache = MambaCache()
        let prefixCache = MambaCache()

        let full = layer(input, cache: fullCache)
        let speculative = layer(input, cache: speculativeCache, checkpointAfter: 1)
        _ = layer(input[0..., ..<1, 0...], cache: prefixCache)
        eval(full, speculative)

        #expect(allClose(speculative, full, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        #expect(speculativeCache.restoreSpeculativeCheckpoint())
        for (actual, reference) in zip(speculativeCache.state, prefixCache.state) {
            eval(actual, reference)
            #expect(
                allClose(actual, reference, rtol: gdnStateTolerance, atol: gdnStateTolerance)
                    .item(Bool.self))
        }
    }

    @Test
    func testQwen35HybridCacheRewindRestoresAttentionAndRecurrentStateAtomically() {
        let attention = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, 3, 2])
        _ = attention.update(keys: keys, values: keys)

        let recurrent = MambaCache()
        let checkpointConv = MLXArray.ones([1, 1, 4])
        let checkpointState = MLXArray.ones([1, 2, 2, 2])
        recurrent.saveSpeculativeCheckpoint(
            convState: checkpointConv, recurrentState: checkpointState, advancedBy: 1)
        recurrent[0] = MLXArray.zeros([1, 1, 4])
        recurrent[1] = MLXArray.zeros([1, 2, 2, 2])

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: 1)
        #expect(rewound == 1)
        #expect(attention.offset == 2)
        #expect(!recurrent.hasSpeculativeCheckpoint)

        let restored = recurrent.state
        #expect(restored.count == 2)
        eval(restored[0], restored[1])
        #expect(allClose(restored[0], checkpointConv, rtol: 0, atol: 0).item(Bool.self))
        #expect(allClose(restored[1], checkpointState, rtol: 0, atol: 0).item(Bool.self))
    }
}

// MARK: - Tape recording and hidden-state capture

// A speculative decoder with tape rollback arms a recording cache for its verification pass
// and reads the outputs of chosen layers. The harm these tests guard against is silent: a
// verification that skips recording, or a capture taken from the wrong layer, corrupts the
// decoding that follows without raising an error.

/// A recurrent cache that counts tape recordings. While armed it stands in for the gated
/// delta update (zeros of the right shapes), which is enough to follow the call path.
private final class TapeSpyCache: MambaCache, GatedDeltaTapeCache {
    var isArmed = false
    private(set) var recordedSteps: [(tape: Int, projectedInput: Int)] = []

    func resetRecords() {
        recordedSteps.removeAll()
    }

    func gatedDeltaWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        (
            MLXArray.zeros(v.shape, dtype: v.dtype), state,
            MLXArray.zeros([v.dim(0), v.dim(1), 1])
        )
    }

    func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray) {
        recordedSteps.append((tape.dim(1), qkv.dim(1)))
    }
}

/// Three recurrent layers followed by one attention layer.
private func makeHybridQwen35(seed: UInt64, tieWordEmbeddings: Bool = true) throws
    -> MLXLLM.Qwen35TextModel
{
    let cfg = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            qwen35TextConfigJSON(
                mtpLayers: 0, hiddenLayers: 4, fullAttentionInterval: 4,
                tieWordEmbeddings: tieWordEmbeddings
            ).utf8))
    // Task-local seed: the global random state is left alone.
    let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
        MLXLLM.Qwen35TextModel(cfg)
    }
    eval(model)
    return model
}

private func tokenRow(_ ids: [Int32]) -> MLXArray {
    MLXArray(ids).reshaped(1, ids.count)
}

@Suite(.serialized)
struct Qwen35TapeRecordingAndCaptureTests {
    @Test
    func testQwen35ArmedTapeCacheRecordsOncePerRecurrentLayer() throws {
        let model = try makeHybridQwen35(seed: 7)
        #expect(model.model.hasLinearLayers)
        let caches: [KVCache?] = try model.newCache(parameters: nil).map {
            $0 is MambaCache ? TapeSpyCache() : $0
        }
        let spies = caches.compactMap { $0 as? TapeSpyCache }
        #expect(spies.count == 3)

        // Disarmed control: an ordinary prefill, then an ordinary single-token step (the
        // compiled whole-step route) and an ordinary capture. Nothing is recorded.
        eval(model.model(tokenRow([1, 2, 3, 4, 5]), cache: caches))
        eval(model.model(tokenRow([6]), cache: caches))
        eval(model.model.callCapturing(tokenRow([7, 8]), cache: caches, captureLayerIDs: [0]).0)
        #expect(spies.allSatisfy { $0.recordedSteps.isEmpty })

        spies.forEach { $0.isArmed = true }

        // Whole-model single-token forward: must leave the compiled whole-step route.
        eval(model.model(tokenRow([9]), cache: caches))
        for spy in spies {
            #expect(spy.recordedSteps.count == 1)
            #expect(spy.recordedSteps.first?.tape == 1)
            #expect(spy.recordedSteps.first?.projectedInput == 1)
            spy.resetRecords()
        }

        // Single-token capture: must leave the compiled single-layer route.
        eval(model.model.callCapturing(tokenRow([10]), cache: caches, captureLayerIDs: [1]).0)
        for spy in spies {
            #expect(spy.recordedSteps.count == 1)
            #expect(spy.recordedSteps.first?.projectedInput == 1)
            spy.resetRecords()
        }

        // Multi-token capture: one tape for the whole block, with the complete projected
        // input of the block.
        eval(
            model.model.callCapturing(
                tokenRow([11, 12, 13]), cache: caches, captureLayerIDs: [1]
            ).0)
        for spy in spies {
            #expect(spy.recordedSteps.count == 1)
            #expect(spy.recordedSteps.first?.tape == 3)
            #expect(spy.recordedSteps.first?.projectedInput == 3)
            spy.resetRecords()
        }

        // Disarmed again: back to no recording.
        spies.forEach { $0.isArmed = false }
        eval(model.model(tokenRow([14]), cache: caches))
        #expect(spies.allSatisfy { $0.recordedSteps.isEmpty })
    }

    @Test
    func testQwen35CapturingMatchesOrdinaryForwardAndCacheState() throws {
        let model = try makeHybridQwen35(seed: 11)
        let tokens = tokenRow([1, 2, 3, 4, 5])
        let ordinaryCaches = try model.newCache(parameters: nil)
        let capturingCaches = try model.newCache(parameters: nil)

        let ordinary = model.model(tokens, cache: ordinaryCaches)
        let (hidden, captured) = model.model.callCapturing(
            tokens, cache: capturingCaches, captureLayerIDs: [0, 3])
        eval(ordinary, hidden)

        #expect(allClose(hidden, ordinary, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        for (capturing, plain) in zip(capturingCaches, ordinaryCaches) {
            #expect(capturing.offset == plain.offset)
            #expect(capturing.state.count == plain.state.count)
            for (a, b) in zip(capturing.state, plain.state) {
                #expect(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
        }

        // Zero-based keys, the selected layers only.
        #expect(Set(captured.keys) == [0, 3])
        let first = try #require(captured[0])
        let last = try #require(captured[3])
        #expect(first.shape == [1, 5, 16])
        // Layer 3 is the last layer: its captured residual, normalized, is the result.
        #expect(allClose(model.model.norm(last), hidden, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        // The first layer's output is a different value, so the keys are not mixed up.
        #expect(!allClose(first, last, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    @Test
    func testQwen35PublicEmbeddingAndHeadReproduceLogits() throws {
        let tokens = tokenRow([1, 2, 3, 4])
        for tieWordEmbeddings in [true, false] {
            let model = try makeHybridQwen35(seed: 13, tieWordEmbeddings: tieWordEmbeddings)
            #expect((model.lmHead == nil) == tieWordEmbeddings)
            #expect(Qwen35DFlashPublicAPIProbe.isHybrid(model))

            let logits = model(tokens, cache: nil)
            let hidden = model.model(tokens, cache: nil)
            let viaPublicAccess = Qwen35DFlashPublicAPIProbe.logits(model, hidden)
            let captureCaches = try model.newCache(parameters: nil)
            let (viaCapture, captured) = Qwen35DFlashPublicAPIProbe.forwardWithCapture(
                model, inputIDs: tokens, cache: captureCaches, captureLayerIDs: [2])
            eval(logits, viaPublicAccess, viaCapture)

            #expect(allClose(viaPublicAccess, logits, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(allClose(viaCapture, logits, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(Set(captured.keys) == [2])
            #expect(
                Qwen35DFlashPublicAPIProbe.embed(model, tokens).shape == [1, 4, 16])
        }

        // The existing fixture default is one full-attention layer: not hybrid.
        let attentionOnly = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(qwen35TextConfigJSON(mtpLayers: 0).utf8))
        #expect(!Qwen35DFlashPublicAPIProbe.isHybrid(MLXLLM.Qwen35TextModel(attentionOnly)))
    }
}

@Suite(.serialized)
struct Qwen35MTPRegistrationTests {
    @Test
    func registrationsCreateTextAndVLMDrafters() async throws {
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35TextConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5_text")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)

        await MLXVLM.Qwen35VLMMTPRegistration.register()

        let wrappedTextModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(wrappedTextModel is MLXLLM.Qwen35MTPDraftModel)

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35StandaloneMTPConfigJSON().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standalone is MLXLLM.Qwen35MTPDraftModel)
        #expect(standalone.maximumBlockSize == 2)
        #expect(standalone.requiresPromptPrefill)
        #expect(!standalone.requiresSharedTargetKV)
        #expect(standalone.requiresGreedySampling)
    }

    @Test
    func registrationsAreOrderIndependentForSharedModelTypes() async throws {
        await MLXVLM.Qwen35VLMMTPRegistration.register()
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let vlmModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35VLMConfigJSON(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(qwen35WrappedTextConfigJSON(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)
    }
}

private func qwen35TextConfigJSON(
    mtpLayers: Int,
    mtpUseDedicatedEmbeddings: Bool = false,
    numExperts: Int = 0,
    hiddenLayers: Int = 1,
    fullAttentionInterval: Int = 1,
    tieWordEmbeddings: Bool = true
) -> String {
    """
    {
      "model_type": "qwen3_5_text",
      "hidden_size": 16,
      "num_hidden_layers": \(hiddenLayers),
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "linear_num_value_heads": 2,
      "linear_num_key_heads": 1,
      "linear_key_head_dim": 8,
      "linear_value_head_dim": 8,
      "linear_conv_kernel_dim": 2,
      "rms_norm_eps": 1e-6,
      "vocab_size": 16,
      "rope_theta": 100000.0,
      "partial_rotary_factor": 0.25,
      "max_position_embeddings": 64,
      "tie_word_embeddings": \(tieWordEmbeddings),
      "attention_bias": false,
      "full_attention_interval": \(fullAttentionInterval),
      "mtp_num_hidden_layers": \(mtpLayers),
      "mtp_use_dedicated_embeddings": \(mtpUseDedicatedEmbeddings),
      "num_experts": \(numExperts),
      "num_experts_per_tok": \(numExperts == 0 ? 0 : 1),
      "moe_intermediate_size": 16,
      "shared_expert_intermediate_size": 16,
      "rope_parameters": {
        "type": "default",
        "rope_theta": 100000.0,
        "partial_rotary_factor": 0.25
      }
    }
    """
}

private func qwen35VLMConfigJSON(mtpLayers: Int) -> String {
    """
    {
      "model_type": "qwen3_5",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: mtpLayers)),
      "vision_config": {
        "model_type": "qwen3_5_vit",
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "out_hidden_size": 16,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "num_position_embeddings": 16
      }
    }
    """
}

private func qwen35WrappedTextConfigJSON(modelType: String) -> String {
    """
    {
      "model_type": "\(modelType)",
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1))
    }
    """
}

private func qwen35StandaloneMTPConfigJSON() -> String {
    """
    {
      "model_type": "qwen3_5_mtp",
      "block_size": 3,
      "text_config": \(qwen35TextConfigJSON(mtpLayers: 1)),
      "tie_word_embeddings": true,
      "vision_config": {}
    }
    """
}
