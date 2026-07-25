//! route.zig — direct / pivot resolution. SPEC §10.
//!
//! Bergamot models are single-direction and one-to-one; non-English pairs have
//! no direct model. Firefox resolves those by pivoting through English, at most
//! once. So do we, and the "at most once" is asserted rather than intended.

const std = @import("std");
const assert = std.debug.assert;

const abi = @import("abi.zig");
const format = @import("model/format.zig");

pub const Kind = enum(i32) {
    none = 0,
    direct = 1,
    pivot = 2,
};

pub const no_slot: u8 = 0xff;

pub const Plan = struct {
    kind: Kind,
    /// The host asked for a pair we can answer without a model. Still a
    /// `direct` route as far as the ABI is concerned.
    identity: bool = false,
    first: u8 = no_slot,
    second: u8 = no_slot,

    pub const none: Plan = .{ .kind = .none };

    pub fn hops(self: Plan) u8 {
        return switch (self.kind) {
            .none => 0,
            .direct => if (self.identity) 0 else 1,
            .pivot => 2,
        };
    }
};

/// `models` is the instance's slot table; unloaded slots are skipped. Language
/// codes are untrusted, so an invalid one is `none`, never an assertion.
pub fn resolve(models: []const format.Model, src: abi.Lang, tgt: abi.Lang) Plan {
    assert(models.len <= abi.limits.models);

    if (!abi.langValid(src) or !abi.langValid(tgt)) return .none;
    if (src == tgt) return .{ .kind = .direct, .identity = true };

    if (find(models, src, tgt)) |slot| {
        return .{ .kind = .direct, .first = slot };
    }

    // SPEC §10: one hop through English, never two.
    if (src != abi.lang_en and tgt != abi.lang_en) {
        if (find(models, src, abi.lang_en)) |a| {
            if (find(models, abi.lang_en, tgt)) |b| {
                assert(a != b);
                const plan: Plan = .{ .kind = .pivot, .first = a, .second = b };
                assert(plan.hops() == 2);
                return plan;
            }
        }
    }

    return .none;
}

fn find(models: []const format.Model, src: abi.Lang, tgt: abi.Lang) ?u8 {
    assert(abi.langValid(src) and abi.langValid(tgt));
    for (models, 0..) |m, i| {
        if (!m.loaded) continue;
        if (m.src_lang == src and m.tgt_lang == tgt) return @intCast(i);
    }
    return null;
}

// -- tests ------------------------------------------------------------------

fn stub(src: *const [2]u8, tgt: *const [2]u8) format.Model {
    var m: format.Model = .empty;
    m.loaded = true;
    m.src_lang = abi.langFrom(src);
    m.tgt_lang = abi.langFrom(tgt);
    return m;
}

test "identity needs no model" {
    const models: [0]format.Model = .{};
    const p = resolve(&models, abi.langFrom("es"), abi.langFrom("es"));
    try std.testing.expectEqual(Kind.direct, p.kind);
    try std.testing.expect(p.identity);
    try std.testing.expectEqual(@as(u8, 0), p.hops());
}

test "direct beats pivot" {
    const models = [_]format.Model{
        stub("es", "de"),
        stub("es", "en"),
        stub("en", "de"),
    };
    const p = resolve(&models, abi.langFrom("es"), abi.langFrom("de"));
    try std.testing.expectEqual(Kind.direct, p.kind);
    try std.testing.expectEqual(@as(u8, 0), p.first);
}

test "pivot composes exactly two hops through English" {
    const models = [_]format.Model{ stub("es", "en"), stub("en", "de") };
    const p = resolve(&models, abi.langFrom("es"), abi.langFrom("de"));
    try std.testing.expectEqual(Kind.pivot, p.kind);
    try std.testing.expectEqual(@as(u8, 0), p.first);
    try std.testing.expectEqual(@as(u8, 1), p.second);
    try std.testing.expectEqual(@as(u8, 2), p.hops());
}

test "no path is a status, not a second pivot" {
    // fr->de would need fr->es->en->de. Never.
    const models = [_]format.Model{ stub("fr", "es"), stub("en", "de") };
    try std.testing.expectEqual(Kind.none, resolve(&models, abi.langFrom("fr"), abi.langFrom("de")).kind);
}

test "junk language codes resolve to none" {
    const models = [_]format.Model{stub("es", "en")};
    try std.testing.expectEqual(Kind.none, resolve(&models, 0, abi.lang_en).kind);
    try std.testing.expectEqual(Kind.none, resolve(&models, 0xffff, 0xffff).kind);
    try std.testing.expectEqual(Kind.none, resolve(&models, abi.langFrom("es"), 'E' << 8 | 'N').kind);
}

test "unloaded slots are invisible" {
    var models = [_]format.Model{ stub("es", "en"), stub("en", "de") };
    models[1].loaded = false;
    try std.testing.expectEqual(Kind.none, resolve(&models, abi.langFrom("es"), abi.langFrom("de")).kind);
}
