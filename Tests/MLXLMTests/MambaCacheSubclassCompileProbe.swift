import MLX
import MLXLMCommon

/// Compile probe for the cache extension boundary.
///
/// This file imports `MLXLMCommon` without `@testable` on purpose, so it sees only what a
/// separate module sees. It stops compiling if `MambaCache` (and with it `ArraysCache`) is no
/// longer `open`, or if `trim` can no longer be overridden.
///
/// The shape is the one a speculative decoder's rollback cache needs: it inherits the storage
/// and sequence metadata, keeps its own snapshot, and replaces `trim`.
final class ExternalRollbackProbeCache: MambaCache {
    private(set) var trimRequests: [Int] = []

    @discardableResult
    override func trim(_ n: Int) -> Int {
        trimRequests.append(n)
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }
}
