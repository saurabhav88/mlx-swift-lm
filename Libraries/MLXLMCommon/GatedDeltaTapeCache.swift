// Derived from SwiftLM (SharpAI, MIT): see ThirdPartyLicenses/SwiftLM-LICENSE.txt.
// Cache-owned GDN verification avoids a dependency from the model to DFlash.
import MLX

public protocol GatedDeltaTapeCache: AnyObject {
    var isArmed: Bool { get }
    func gatedDeltaWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray, state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray)
    func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray)
}
