# fp32 + `tdm_global_load` produces `-nan` output — regression, not a new gap

**Status: FIXED, root-caused (`3d8f3ea`).** Bisect identified commit `16bbbfa` ("gfx1250
WMMA: implement 11-item prioritized action plan from hardware review") as the breaking
commit. Its COR-001 item enforced `lds_double_buffer=1` for all fp32 WMMA configs —
including TDM. But TDM's `tensor_load_to_lds` writes to a fixed LDS base address in SGPRs
(`s_tdm_g0(1)`/`s_tdm_g0_b(1)`), set once in the prologue and invisible to
`wmma_main_loop.py`'s generic VGPR-only `emit_buffer_switch()`. With double-buffering,
TDM always wrote to buffer 0 while the read offsets alternated to buffer 1 every other
iteration, reading uninitialized LDS and producing silent `-nan` output. Fix: added a
`buffer_switch_extra_functor` callback to `ctrl_wmma_main_loop_t`, called by
`emit_buffer_switch()` and the prologue's initial buffer advance, which XORs the TDM
descriptor's SGPR LDS base with `lds_single_size` in lockstep with the VGPR toggles.
Hardware-validated: fwd/bwd `_tdm` and `_tdm_direct`, 3 shapes each, all `valid:y`;
non-TDM fp32 kernels in the master config still `valid:y`.

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

## Resolution (bisect + root cause + fix)

### Bisect

Good anchor: `35dd4ab` (Phase 29, "TDM single-issuer-wave fix") — confirmed `valid:y` on
the exact Phase 29 shape (`n8 c2048 32x32 k2048 1x1`), 10/10 runs, fresh build.
Bad anchor: `ffa4577` (HEAD at time of investigation) — confirmed `pred:-nan`, 10/10 runs.
`git bisect` across 51 commits touching the 5 relevant source paths identified:

**Breaking commit: `16bbbfa`** ("gfx1250 WMMA: implement 11-item prioritized action plan
from hardware review"). The commit immediately before it (`a1b8889`, "architecture map")
is `valid:y`; `16bbbfa` is `pred:-nan`.

### Root cause

Commit `16bbbfa`'s COR-001 item enforced `lds_double_buffer=1` for all fp32 WMMA tunables
(a structural assert in `igemm_base.py`), and "repaired" all fp32 config files missing the
flag — including the TDM configs. This was correct for non-TDM fp32 (COR-001's original
purpose: fp32's 4-byte-wide WMMA operands need double-buffering to avoid a last-lane
LDS-visibility race at high occupancy). But it broke TDM:

TDM's `tensor_load_to_lds` instruction writes tile data directly to LDS using a **tensor
descriptor** whose LDS base address is a fixed SGPR constant (`s_tdm_g0(1)` for operand A,
`s_tdm_g0_b(1)` for B), set once in the prologue by `_emit_tdm_descriptor_setup_a/b`.
`wmma_main_loop.py`'s `emit_buffer_switch()` toggles only the VGPR read/store offsets
(`v_sst_a_os`, `v_sld_a_os`, `v_sld_b_os`) via XOR with `lds_single_size` — it cannot
toggle the TDM descriptor's SGPR LDS base. With double-buffering enabled, TDM always wrote
to buffer 0 while the read offsets alternated to buffer 1 every other iteration, reading
uninitialized LDS and producing `-nan`.

fp16/bf16 TDM were unaffected because COR-001's `lds_double_buffer=1` mandate only targets
`precision == 'fp32'`; fp16/bf16 TDM configs stayed single-buffered (the default). The
regression doc's original hypothesis about K-tail/stride/element-size assumptions was wrong
— the actual divergence was in the buffer-switching layer, not in TDM's shared plumbing.

### Fix (`3d8f3ea`)

Added a `buffer_switch_extra_functor` field to `ctrl_wmma_main_loop_t`
(`wmma_main_loop.py`), called by `emit_buffer_switch()` and the prologue's initial buffer
advance, right after the VGPR XOR toggles. The fwd/bwd generators set this functor when
`tdm_global_load=1` AND `lds_buffer_num==2`, to emit:

```asm
s_xor_b32 s[s_tdm_g0+1], lds_single_size, s[s_tdm_g0+1]   ; TDM A descriptor LDS base toggle
s_xor_b32 s[s_tdm_g0_b+1], lds_single_size, s[s_tdm_g0_b+1]   ; TDM B descriptor LDS base toggle
```

This makes TDM's `tensor_load_to_lds` write to the same buffer `v_sst_a_os` points at,
maintaining the double-buffer invariant (store and read always target different buffers).

### Hardware validation

All runs on real gfx1250 silicon, `-V 1` verify:

| Direction | Config | n128 c1024 17x17 k1024 | n64 c512 28x28 k512 | n8 c2048 32x32 k2048 |
|-----------|--------|------------------------|---------------------|----------------------|
| fwd | `_tdm` | valid:y | valid:y | valid:y |
| fwd | `_tdm_direct` | valid:y | valid:y | valid:y |
| bwd | `_tdm` | valid:y | valid:y | valid:y |
| bwd | `_tdm_direct` | valid:y | valid:y | valid:y |

Non-TDM fp32 kernels in `config/igemm_fwd_gtc_gfx1250_nhwc_fp32_all.config` (spot-checked
10+ kernels including `_dbuf`, `_dbuf_async`, `_dbuf_direct`, `_dbuf_gkgs`) all still
`valid:y` — no regression from the fix.

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
