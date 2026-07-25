//! The native test entry point.
//!
//! It sits at the repository root rather than under `test/` because a Zig
//! module may only import files below its root file's directory, and the suite
//! needs to reach both `src/` and `test/`. Every module in `src/` is referenced
//! here so that `zig build test` runs the whole tree.

test {
    _ = @import("src/root.zig");
    _ = @import("src/abi.zig");
    _ = @import("src/arena.zig");
    _ = @import("src/route.zig");
    _ = @import("src/runtime.zig");
    _ = @import("src/model/format.zig");
    _ = @import("src/model/layout.zig");
    _ = @import("src/model/names.zig");
    _ = @import("src/model/repack.zig");
    _ = @import("src/kernel/math.zig");
    _ = @import("src/kernel/backend.zig");
    _ = @import("src/graph/pass.zig");
    _ = @import("src/graph/encoder.zig");
    _ = @import("src/graph/decoder.zig");
    _ = @import("src/graph/attention.zig");
    _ = @import("src/graph/shortlist.zig");
    _ = @import("src/tok/ssplit.zig");
    _ = @import("src/tok/trie.zig");
    _ = @import("src/tok/unigram.zig");

    _ = @import("test/runtime_test.zig");
    _ = @import("test/artifact.zig");
    _ = @import("test/differential_test.zig");
    _ = @import("test/golden/vectors_test.zig");
    _ = @import("test/golden/load_test.zig");
    _ = @import("test/golden/translate_test.zig");
    _ = @import("test/fuzz/tokenizer_test.zig");
}
