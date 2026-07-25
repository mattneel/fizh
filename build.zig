//! fizh build.
//!
//! Two wasm artifacts, one native test binary, and the mechanical half of
//! SPEC §12 enforced as build steps.
//!
//!   zig build wasm    fizh.baseline.wasm + fizh.relaxed.wasm into zig-out/wasm
//!   zig build test    native differential + unit tests (T0/T1/T2)
//!   zig build tiger   Tiger Style greps over src/ (SPEC §12)
//!   zig build check   wasm budget audit: imports, exports, gzipped size (SPEC §3)
//!   zig build ci      all of the above

const std = @import("std");

/// SPEC §3. Listed explicitly rather than inherited from the `generic` wasm CPU
/// model so that the artifact's feature set is exactly what the SPEC says and
/// a toolchain bump cannot silently widen it.
const baseline_features = [_]std.Target.wasm.Feature{
    .simd128,
    .bulk_memory,
    .sign_ext,
    .nontrapping_fptoint,
    .mutable_globals,
};

const relaxed_features = baseline_features ++ [_]std.Target.wasm.Feature{.relaxed_simd};

/// SPEC §3 budgets.
const max_gzipped_bytes: u32 = 200 * 1024;
const max_exports: u32 = 10;
const max_imports: u32 = 0;

pub fn build(b: *std.Build) void {
    const native_target = b.standardTargetOptions(.{});
    // SPEC §3: the ship build is ReleaseSafe. Assertions stay on.
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseSafe });

    // ---- wasm artifacts ---------------------------------------------------

    const baseline = addWasm(b, "fizh.baseline", &baseline_features);
    const relaxed = addWasm(b, "fizh.relaxed", &relaxed_features);

    const install_baseline = b.addInstallArtifact(baseline, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    });
    const install_relaxed = b.addInstallArtifact(relaxed, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    });

    // SPEC §3: "Host picks via probe-module feature detection." Written out by
    // hand rather than compiled — see ADR 0006 and tools/make_probe.zig.
    const make_probe = b.addExecutable(.{
        .name = "make-probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/make_probe.zig"),
            .target = native_target,
            .optimize = .Debug,
        }),
    });
    const run_make_probe = b.addRunArtifact(make_probe);
    const probe_bin = run_make_probe.addOutputFileArg("fizh.probe.wasm");
    const install_probe = b.addInstallFileWithDir(probe_bin, .{ .custom = "wasm" }, "fizh.probe.wasm");

    const wasm_step = b.step("wasm", "Build both wasm artifacts and the feature probe");
    wasm_step.dependOn(&install_baseline.step);
    wasm_step.dependOn(&install_relaxed.step);
    wasm_step.dependOn(&install_probe.step);
    b.getInstallStep().dependOn(wasm_step);

    // ---- native tests -----------------------------------------------------

    const test_mod = b.createModule(.{
        .root_source_file = b.path("test.zig"),
        .target = native_target,
        .optimize = optimize,
    });
    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    run_tests.has_side_effects = true;

    const test_step = b.step("test", "Run the native test suite");
    test_step.dependOn(&run_tests.step);

    // ---- Tiger Style (SPEC §12) ------------------------------------------

    const tiger = b.addExecutable(.{
        .name = "tiger",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/tiger.zig"),
            .target = native_target,
            .optimize = .Debug,
        }),
    });
    const run_tiger = b.addRunArtifact(tiger);
    run_tiger.addDirectoryArg(b.path("src"));
    run_tiger.has_side_effects = true;

    const tiger_step = b.step("tiger", "Enforce the mechanically checkable half of SPEC §12");
    tiger_step.dependOn(&run_tiger.step);

    // ---- wasm budget audit (SPEC §3) -------------------------------------

    const auditor = b.addExecutable(.{
        .name = "wasm-audit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/wasm_audit.zig"),
            .target = native_target,
            .optimize = .Debug,
        }),
    });

    const check_step = b.step("check", "Audit the wasm artifacts against the SPEC §3 budgets");
    for ([_]*std.Build.Step.Compile{ baseline, relaxed }) |artifact| {
        const run_audit = b.addRunArtifact(auditor);
        run_audit.addFileArg(artifact.getEmittedBin());
        run_audit.addArgs(&.{
            b.fmt("--max-gzip={d}", .{max_gzipped_bytes}),
            b.fmt("--max-exports={d}", .{max_exports}),
            b.fmt("--max-imports={d}", .{max_imports}),
        });
        run_audit.has_side_effects = true;
        check_step.dependOn(&run_audit.step);
    }

    const run_probe_audit = b.addRunArtifact(auditor);
    run_probe_audit.addFileArg(probe_bin);
    run_probe_audit.addArgs(&.{ "--max-gzip=4096", "--max-imports=0", "--require-relaxed" });
    run_probe_audit.has_side_effects = true;
    check_step.dependOn(&run_probe_audit.step);

    // ---- converter agreement ---------------------------------------------
    //
    // tools/convert.py and src/model/format.zig are two implementations of
    // SPEC §6. This step runs one against the other.

    const fzm_load = b.addExecutable(.{
        .name = "fzm-load",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fzm_load.zig"),
            .target = native_target,
            .optimize = .Debug,
            .imports = &.{.{
                .name = "fizh",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/runtime.zig"),
                    .target = native_target,
                    .optimize = .Debug,
                }),
            }},
        }),
    });
    b.installArtifact(fzm_load);

    const selftest_out = "zig-out/selftest.fzm";
    const run_convert = b.addSystemCommand(&.{ "python3", "tools/convert.py", "--selftest", selftest_out });
    run_convert.has_side_effects = true;

    const run_load = b.addRunArtifact(fzm_load);
    run_load.addArgs(&.{
        selftest_out,     "--d-model", "32",     "--ffn",         "64",
        "--vocab",        "64",        "--enc",  "2",             "--dec",
        "1",              "--heads",   "2",      "--slot-bytes",  "1048576",
        "--src-tokens",   "32",        "--tgt-tokens", "48",
    });
    run_load.has_side_effects = true;
    run_load.step.dependOn(&run_convert.step);

    const convert_step = b.step("convert-selftest", "Check tools/convert.py against the real loader");
    convert_step.dependOn(&run_load.step);

    // ---- perf harness (SPEC §13 T5, SPEC §14) -----------------------------

    const bench = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/bench.zig"),
            .target = native_target,
            // The ship build is ReleaseSafe (SPEC §3); measuring anything else
            // by default would measure a binary nobody runs.
            .optimize = .ReleaseSafe,
            .imports = &.{.{
                .name = "fizh",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/runtime.zig"),
                    .target = native_target,
                    .optimize = .ReleaseSafe,
                }),
            }},
        }),
    });
    b.installArtifact(bench);

    // A synthetic artifact has random weights, so it never emits `</s>`: every
    // sentence runs to the step limit and the §14 timings measure the bound
    // rather than the workload. Prefer a real model whenever one has been
    // fetched; fall back to the synthetic one only so `zig build bench` works
    // in a bare checkout, and say so in the step name. ADR 0016.
    const synth_model = "zig-out/bench.fzm";
    const build_bench_model = b.addSystemCommand(&.{
        "python3", "tools/convert.py", "--selftest", "--big", synth_model,
    });
    build_bench_model.has_side_effects = true;

    const real_model = "zig-out/esen.fzm";
    const have_real = if (b.build_root.handle.access(b.graph.io, real_model, .{})) |_| true else |_| false;
    const bench_model = if (have_real) real_model else synth_model;

    const run_bench = b.addRunArtifact(bench);
    run_bench.addArg(bench_model);
    if (b.args) |extra| run_bench.addArgs(extra);
    run_bench.has_side_effects = true;

    if (!have_real) run_bench.step.dependOn(&build_bench_model.step);

    const bench_step = b.step("bench", if (have_real)
        "SPEC §14 budgets against a real Bergamot artifact"
    else
        "SPEC §14 budgets against a synthetic artifact (run tools/fetch-model.sh for real timings)");
    bench_step.dependOn(&run_bench.step);

    // SPEC §3: "Measure the ReleaseFast delta at M7; the number goes in an ADR
    // either way." This is the step that produces the number.
    const bench_fast = b.addExecutable(.{
        .name = "bench-fast",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/bench.zig"),
            .target = native_target,
            .optimize = .ReleaseFast,
            .imports = &.{.{
                .name = "fizh",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/runtime.zig"),
                    .target = native_target,
                    .optimize = .ReleaseFast,
                }),
            }},
        }),
    });
    const run_bench_fast = b.addRunArtifact(bench_fast);
    run_bench_fast.addArg(bench_model);
    run_bench_fast.has_side_effects = true;
    if (!have_real) run_bench_fast.step.dependOn(&build_bench_model.step);

    const bench_fast_step = b.step("bench-fast", "The same benchmark in ReleaseFast, for the SPEC §3 delta");
    bench_fast_step.dependOn(&run_bench_fast.step);

    // ---- the oracle (SPEC §13 T0 and T2) ----------------------------------

    const trace = b.addExecutable(.{
        .name = "trace",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/trace.zig"),
            .target = native_target,
            .optimize = .ReleaseSafe,
            .imports = &.{.{
                .name = "fizh",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/runtime.zig"),
                    .target = native_target,
                    .optimize = .ReleaseSafe,
                }),
            }},
        }),
    });
    b.installArtifact(trace);

    const trace_out = "zig-out/trace.bin";
    const oracle_text = "hola aaaa baaa caaa";

    const run_trace = b.addRunArtifact(trace);
    run_trace.addArgs(&.{ bench_model, oracle_text, trace_out });
    run_trace.has_side_effects = true;
    run_trace.step.dependOn(&build_bench_model.step);

    const run_oracle = b.addSystemCommand(&.{
        "python3", "tools/reference.py", "--compare", trace_out, "--model", bench_model,
    });
    run_oracle.setEnvironmentVariable("PYTHONPATH", "tools");
    run_oracle.has_side_effects = true;
    run_oracle.step.dependOn(&run_trace.step);

    // The bound `stage_profile` uses for the later layers is only defensible if
    // the noise floor is measured, so CI measures it.
    const run_selfcheck = b.addSystemCommand(&.{
        "python3", "tools/reference.py", "--selfcheck", "--model", bench_model,
    });
    run_selfcheck.setEnvironmentVariable("PYTHONPATH", "tools");
    run_selfcheck.has_side_effects = true;
    run_selfcheck.step.dependOn(&build_bench_model.step);

    const oracle_step = b.step("oracle", "SPEC §13 T2: full-model differential against tools/reference.py");
    oracle_step.dependOn(&run_oracle.step);
    oracle_step.dependOn(&run_selfcheck.step);

    const regen_golden = b.addSystemCommand(&.{
        "python3", "tools/reference.py", "--golden", "test/golden/vectors.zig",
    });
    regen_golden.setEnvironmentVariable("PYTHONPATH", "tools");
    regen_golden.has_side_effects = true;
    const golden_step = b.step("golden", "Regenerate the SPEC §13 T0 vectors from the oracle");
    golden_step.dependOn(&regen_golden.step);

    // ---- quality (SPEC §13 T4) --------------------------------------------

    const translate_exe = b.addExecutable(.{
        .name = "translate",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/translate.zig"),
            .target = native_target,
            .optimize = .ReleaseSafe,
            .imports = &.{.{
                .name = "fizh",
                .module = b.createModule(.{
                    .root_source_file = b.path("src/runtime.zig"),
                    .target = native_target,
                    .optimize = .ReleaseSafe,
                }),
            }},
        }),
    });
    b.installArtifact(translate_exe);

    // The scorer is checked on every CI run; the *scores* need a real Bergamot
    // artifact, which this repository does not have (ADR 0005).
    const run_chrf = b.addSystemCommand(&.{ "python3", "tools/eval/chrf.py" });
    run_chrf.has_side_effects = true;

    // A step that emits a *number* gates on a real fetched artifact. Synthetic
    // models have broken a measurement gate twice: the whole suite passed with
    // the wrong decoder architecture (ADR 0008), and §14 timed a decode loop
    // that terminated at step 0 (ADR 0016). Both times the number looked fine.
    // So `eval` and `bench` run against a real model or they are not
    // measurements, and neither is in `ci` — see the `ci` step below.
    const run_eval = b.addSystemCommand(&.{ "python3", "tools/eval/run.py" });
    run_eval.addArgs(&.{ "--model", bench_model, "--src", "es", "--tgt", "en" });
    if (b.args) |extra| run_eval.addArgs(extra);
    run_eval.has_side_effects = true;
    if (!have_real) run_eval.step.dependOn(&build_bench_model.step);
    run_eval.step.dependOn(b.getInstallStep());

    const eval_step = b.step("eval", if (have_real)
        "SPEC §13 T4: chrF++ per corpus, reported separately"
    else
        "SPEC §13 T4 -- NOT A MEASUREMENT without a real model (tools/fetch-model.sh)");
    eval_step.dependOn(&run_chrf.step);
    eval_step.dependOn(&run_eval.step);

    // ---- host end-to-end (SPEC §4, §9, §10) -------------------------------

    const run_host = b.addSystemCommand(&.{ "node", "tools/host.mjs", "zig-out/wasm", selftest_out });
    run_host.has_side_effects = true;
    run_host.step.dependOn(wasm_step);
    run_host.step.dependOn(&run_convert.step);

    const host_step = b.step("host", "Drive the shipped wasm from a JS host, as the PWA would");
    host_step.dependOn(&run_host.step);

    // ---- the real model ---------------------------------------------------
    //
    // Skips cleanly when the model is absent; it is 19 MB of CC-BY-SA data that
    // this repository does not vendor. `tools/fetch-model.sh es en` gets it.

    const run_regress = b.addSystemCommand(&.{ "python3", "tools/regress.py" });
    run_regress.has_side_effects = true;
    run_regress.step.dependOn(b.getInstallStep());

    const real_step = b.step("real", "Check fizh against recorded output of a real Bergamot model");
    real_step.dependOn(&run_regress.step);

    // ---- convenience ------------------------------------------------------

    // `eval` and `bench` are deliberately absent: they emit numbers, and a
    // number from a synthetic artifact is not a measurement (ADR 0016). `real`
    // is here because it gates on a fetched model and skips cleanly without one.
    const ci_step = b.step("ci", "wasm + test + tiger + check + convert-selftest + host + oracle");
    ci_step.dependOn(wasm_step);
    ci_step.dependOn(test_step);
    ci_step.dependOn(tiger_step);
    ci_step.dependOn(check_step);
    ci_step.dependOn(convert_step);
    ci_step.dependOn(host_step);
    ci_step.dependOn(oracle_step);
    ci_step.dependOn(&run_chrf.step);
    ci_step.dependOn(real_step);
}

fn wasmTarget(b: *std.Build, features: []const std.Target.wasm.Feature) std.Build.ResolvedTarget {
    var set = std.Target.Cpu.Feature.Set.empty;
    for (features) |f| set.addFeature(@intFromEnum(f));
    return b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
        .cpu_model = .{ .explicit = &std.Target.wasm.cpu.mvp },
        .cpu_features_add = set,
    });
}

fn addWasm(
    b: *std.Build,
    name: []const u8,
    features: []const std.Target.wasm.Feature,
) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = wasmTarget(b, features),
        // SPEC §3. The ReleaseFast delta gets measured at M7; until then the
        // shipped mode is the only mode.
        .optimize = .ReleaseSafe,
        .strip = true,
        .single_threaded = true,
        .stack_check = false,
    });

    const exe = b.addExecutable(.{ .name = name, .root_module = mod });
    // A reactor, not a command: the host drives it through the exported ABI.
    exe.entry = .disabled;
    exe.rdynamic = true;
    return exe;
}
