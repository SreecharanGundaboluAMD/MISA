# fp32 + `tdm_global_load` produces `-nan` output — regression, not a new gap

**Status: unresolved, not bisected, not mitigated.** Every `tdm_global_load=1` fp32 kernel
in the repo (fwd and bwd; wrw's fp32 master doesn't even build — see the separate, unrelated
finding at the bottom) produces silent `-nan` output on every shape tested, including the
exact shape `docs/gfx1250_wmma_layout.md`'s own Phase 29 recorded as hardware-validated
`valid:y`. This is a **regression somewhere between Phase 29 and current `HEAD`**, not an
unvalidated new code path — the bug is real and currently reachable through the fp32 master
`_all.config` for both fwd and bwd; no driver-side rejection currently excludes it.

## Discovery context

Found incidentally while hardware-validating `docs/gfx1250_wmma_perf_report_v2.md` §8 item 10
(temporal hints, `th:TH_LOAD_RT_NT`/`th:TH_STORE_NT`). That change does **not** touch TDM's
load path at all (TDM uses `tensor_load_to_lds`-style descriptors, not the plain
`global_load_dwordx4` emitters item 10 modified) and does not explain this — confirmed via
`git stash`: the failure reproduces identically on unmodified `HEAD`, with or without the
item 10 change applied.

## Symptom

```
$ python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_fp32_all.config -d /tmp/x
$ ./conv_driver.exe conv -n 128 -c 1024 -H 17 -W 17 -k 1024 -y 1 -x 1 -p 0 -q 0 -F 1 -V 1 \
    --in_layout NHWC --fil_layout NHWC --out_layout NHWC
[fwd:21] igemm_fwd_gtcw_..._dbuf_tdm, cost:1.250ms, tflops:62.072(86.10%) invalid float at 0, ref:2.131315, pred:-nan
[fwd:22] igemm_fwd_gtcw_..._dbuf_tdm_direct, cost:1.255ms, ... pred:-nan
[fwd:23] igemm_fwd_gtcw_..._dbuf_tdm_mtail_ntail, cost:1.249ms, ... pred:-nan
```

Of the 23 tunables the fp32 fwd master searches for this shape, **exactly the 3 whose name
contains `_tdm`** fail; the other 20 (including `_mtail`, `_ntail`, `_mtail_direct`,
`_ntail_direct`, `_direct`, `_saddr`, `_async`, `_gkgs`, `_ktail`, `_mtail_ntail_ktail`,
`_lp2` — every other mechanism combination in the master) are `valid:y`. This isolates the
bug to `tdm_global_load=1` specifically, not some broader fp32 issue.

## Confirmed scope

- **Both fwd and bwd fp32 TDM fail.** `igemm_bwd_gtc_gfx1250_nhwc_fp32_all.config`'s
  `_dbuf_tdm`/`_dbuf_tdm_direct` kernels reproduce the identical `-nan` symptom on the same
  shape (`n128 c1024 17x17 k1024 1x1`, `-F 2`).
- **fp16 TDM does NOT fail** — `igemm_fwd_gtc_gfx1250_nhwc_fp16_tdm.config`'s `_tdm` kernel is
  `valid:y` on both `n128 c1024 17x17 k1024 1x1` and `n8 c2048 32x32 k2048 1x1`. This isolates
  the regression to **fp32 specifically**, not a general TDM breakage — fp16/bf16 TDM users
  are unaffected.
- **Not shape-specific.** Reproduced on 3 different shapes:
  - `n128 c1024 17x17 k1024 1x1` (this report's standing regression shape)
  - `n64 c512 28x28 k512 1x1`
  - `n8 c2048 32x32 k2048 1x1` — **this is the exact shape
    `docs/gfx1250_wmma_layout.md` Phase 29 (2026-08-27) recorded as hardware-validated
    `valid:y` for fp32 TDM** ("Hardware validation: bf16/fp16/fp32 all `valid:y` across the
    exact-multiple large shape (n=8,c=k=2048,H=W=32)..."). It fails on current `HEAD`. This
    is the strongest evidence this is a **regression**, not a gap that was never actually
    validated — something changed fp32's TDM path (or something TDM's fp32 path depends on)
    between Phase 29's commit and now.

## Not yet done (next steps for whoever picks this up)

1. **Bisect** `git log` between Phase 29's commit (search `docs/gfx1250_wmma_layout.md` for
   the exact hash near its "Phase 29" heading) and current `HEAD` for the change that broke
   this — not attempted in this session (out of scope for what was being validated).
2. Given fp16/bf16 TDM still work and only fp32 broke, look first at fp32-specific TDM
   surface area: `python/igemm/igemm_fwd_gtc_wmma_nhwc.py`'s `_emit_tdm_descriptor_setup_a`
   and any `precision`-conditional branch in it or in `tdm_global_to_lds_a` wiring
   (`wmma_main_loop.py`), and fp32's distinct `gemm_k_per_block=4`/`inst_wmma.k=4` shape
   (every other precision uses a wider K) — a K-tail, stride, or element-size assumption
   baked in for the wider fp16/bf16 K might silently mis-handle fp32's narrower one.
3. `-nan`, not a crash or `valid:n`-without-diagnostic — the kernel dispatches and completes,
   so this reads like a genuine uninitialized/garbage LDS or VGPR read (same failure
   signature category as R7's bwd `dbuf`+`lds_row_pad` `-nan`, `docs/gfx1250_bwd_dbuf_ldsrp_nan.md`
   — unrelated mechanism, but worth checking with the same "read `-nan`-producing address
   computation line-by-line against a working precision" method that found R7's issue... except
   here fp16/bf16 TDM's own address computation is presumably shared code with fp32's, so the
   divergence is more likely in a precision-conditional branch than in TDM's shared plumbing.
4. **Do not add a `tunable_is_valid` rejection speculatively** — no fp32 TDM config is
   currently in wide use or a documented recommendation, so unlike R5/R7 there's no urgent
   need to guard a reachable "recommended" path. But this should be tracked so it doesn't get
   rediscovered from scratch, and any future perf work involving fp32 TDM should re-run this
   repro first.

## Reproduction

```
$ python3 igemm_codegen.py config/igemm_fwd_gtc_gfx1250_nhwc_fp32_all.config -d /tmp/x
$ IGEMM_RUN_ONLY_KERNEL=igemm_fwd_gtcw_nhwc_fp32_bx0_ex0_bt128x128x4_wt16x16_wr4x4_ta1x4x1x1_1x1x1x128_tb1x4x1x1_1x1x1x128_dbuf_tdm \
  ./conv_driver.exe conv -n 8 -c 2048 -H 32 -W 32 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 \
  -l 1 -j 1 -g 1 -F 1 -V 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC
# Expected (bug present): "invalid float ..., pred:-nan"
```

## Separate, unrelated finding discovered while checking this bug's breadth

`config/igemm_wrw_gtc_gfx1250_nhwc_fp32_all.config` **fails to build at all** — not a TDM or
`-nan` issue, a kernel-name collision:

```
$ python3 igemm_codegen.py config/igemm_wrw_gtc_gfx1250_nhwc_fp32_all.config -d /tmp/x
error: symbol 'igemm_wrw_gtcw_..._bt128x128x4_..._dbuf_gkgs.kd' is already defined
error: symbol 'igemm_wrw_gtcw_..._bt64x64x4_..._dbuf_gkgs' is already defined
```

Two distinct tunable sections (128x128 and 64x64 tiles, both `gemm_k_per_block=4`) mangle to
kernel names that collide — the tile dimensions aren't folded into the `_dbuf_gkgs` name the
way every other suffix combination is. This means **the wrw fp32 master config is entirely
unbuildable today**, independent of the TDM bug above. Not investigated further (out of
scope for this discovery pass) — flagged here so it isn't lost. Likely fix location:
`igemm_gtc_encode_kernel_name` in `python/igemm/igemm_base.py` (fp16/bf16 don't hit this
because their corresponding sections apparently differ in some other name-affecting field
that fp32's don't — not yet identified which).
