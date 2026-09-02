# Codegen Comments Analysis Report

**Directory:** `python/codegen/`
**Task:** Extract and classify all non-license comments from every Python source file.
**License header:** Lines 1-24 (MIT License block) are boilerplate and excluded from analysis.

---

## Summary Table

| File | Total Comments | Remove | Compress | Preserve | % Removable |
|------|---------------|--------|----------|----------|-------------|
| `__init__.py` | 0 | 0 | 0 | 0 | N/A |
| `amdgpu.py` | 18 | 3 | 0 | 15 | ~17% |
| `compile.py` | 13 | 2 | 0 | 11 | ~15% |
| `config_parser.py` | 9 | 3 | 0 | 6 | ~33% |
| `instruction.py` | 1 | 0 | 0 | 1 | 0% |
| `macro.py` | 3 | 0 | 0 | 3 | 0% |
| `mbb.py` | 1 | 1 | 0 | 0 | 100% |
| `mc.py` | 10 | 1 | 0 | 9 | ~10% |
| `scheduler.py` | 40 | 0 | 0 | 40 | 0% |
| `symbol.py` | 0 | 0 | 0 | 0 | N/A |
| **Total** | **95** | **10** | **0** | **85** | **~11%** |

---

## Per-File Analysis

---

### 1. `__init__.py`

**Total non-license comments:** 0

No comments beyond the license header. File contains only imports.

**Estimated % removable:** N/A

**Preserved comments:** None

---

### 2. `amdgpu.py`

**Total non-license comments:** 18

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 25 | `# pylint: disable=maybe-no-member` | **Preserve** | Linter directive; must remain |
| 26 | `# pylint: disable=maybe-no-member` (continued) | **Preserve** | Linter directive |
| 179 | `# TODO: int4 is half byte` | **Preserve** | TODO marker |
| 202 | `# https://llvm.org/docs/AMDGPUOperandSyntax.html#s` | **Preserve** | Reference URL documenting hardware-specific encoding logic |
| 224 | `# in byte` | **Remove** | Obvious from variable name `lds_size` and context |
| 237 | `# read write` | **Compress** → `# R/W` | Slightly verbose; or Remove since context is clear |
| 283 | `# read write` | **Compress** → `# R/W` | Same as line 237 |
| 397 | `# other sgpr related to be implemented` | **Preserve** | Effectively a TODO (unimplemented feature marker) |
| 417 | `# VCC, FLAT_SCRATCH and XNACK must be counted` | **Preserve** | Hardware-specific architectural rationale for SGPR counting |
| 426 | `# 0-cu mode, 1-wgp mode. for gfx>10` | **Preserve** | Hardware-specific documentation of enum values |
| 448 | `# other sgpr related to be implemented` | **Preserve** | TODO marker (unimplemented feature) |
| 467 | `http://llvm.org/docs/AMDGPUUsage.html#code-object-v3-metadata-mattr-code-object-v3` (in docstring) | **Preserve** | Reference URL in docstring |
| 510 | `# other sgpr related to be implemented` | **Preserve** | TODO marker |
| 529 | `# v3 is 64 byte rodata for each kerenl` | **Preserve** | Hardware-specific code object v3 detail |
| 551 | `# .amdhsa_ieee_mode / .amdhsa_dx10_clamp are rejected outright by the assembler` | **Preserve** | Hardware-specific assembler behavior (gfx1170+) |
| 552 | `# on gfx1170+ (verified with llvm-mc), gfx1250 included.` | **Preserve** | Hardware-specific verification note |
| 571 | `# gfx1250's assembler rejects this directive outright (verified with llvm-mc);` | **Preserve** | Hardware-specific assembler rejection note |
| 572 | `# it's still accepted on gfx1030/gfx11xx/gfx1200.` | **Preserve** | Hardware-specific compatibility note |
| 592 | `# default set to 8` | **Remove** | Obvious default; the `8` literal is self-documenting |
| 595 | `# hard code to 0` | **Remove** | Restates the `0` literal on the same line |

**Estimated % removable:** ~17% (3 of 18: lines 224, 592, 595)

**Preserved comments:** Lines 25, 179, 202, 397, 417, 426, 448, 467, 510, 529, 551, 552, 571, 572

---

### 3. `compile.py`

**Total non-license comments:** 13

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 35 | `# hipclang perfer use hipcc to compile host code` | **Preserve** | Design rationale for flag choice |
| 51 | `# make sure mc output is closed` | **Preserve** | Design note on prerequisite (though the actual call on line 52 is commented out) |
| 52 | `# self.mc.close()` | **Preserve** | Commented-out code |
| 66 | `# print("[hip] " + " ".join(cmd))` | **Preserve** | Commented-out debug/logging code |
| 89 | `# make sure mc output is closed` | **Preserve** | Design note on prerequisite |
| 104 | `# TODO: current compiler treat cov3 as default, so no need add extra flag` | **Preserve** | TODO marker |
| 108 | `# print("[asm] " + " ".join(cmd))` | **Preserve** | Commented-out debug/logging code |
| 142 | `# cmd += ['>', '{}'.format(self.target_disass)]` | **Preserve** | Commented-out code |
| 143 | `# print("[dis] " + " ".join(cmd))` | **Preserve** | Commented-out debug/logging code |
| 200 | `# for multiple files` | **Remove** | Obvious from `type(self.host_cpp) is list` check |
| 212 | `# from `/opt/rocm/bin/hipconfig --cpp_config`` | **Preserve** | Provenance note for derived compiler flags |
| 226 | `# for multiple files` | **Remove** | Obvious from `type(self.host_cpp) is list` check |
| 235 | `# print("[host] " + " ".join(cmd))` | **Preserve** | Commented-out debug/logging code |

**Estimated % removable:** ~15% (2 of 13: lines 200, 226)

**Preserved comments:** Lines 35, 51, 52, 66, 89, 104, 108, 142, 143, 212, 235

---

### 4. `config_parser.py`

**Total non-license comments:** 9

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 77 | `# return a list of section, each section is key-value pair` | **Remove** | Restates function name and return type |
| 87 | (inline: `if line[0] == '#' or line[0] == ';':`) | N/A | This is code, not a comment |
| 91 | `if '#' in line:` | N/A | This is code, not a comment |
| 92 | `return line.split('#')[0]` | N/A | This is code, not a comment |
| 125 | `# [x,x,x,x]` | **Preserve** | Documents the value-list syntax format |
| 137 | `# ( [start], end, [step] )` | **Preserve** | Documents the value-range syntax format |
| 169 | `# TODO: recursive dict not supported` | **Preserve** | TODO marker |
| 218 | `# safe to recursively call here, to better create a value type` | **Remove** | Restates what the code does (recursive call is visible) |
| 222 | `# finaly return string` | **Remove** | Restates the `return value` statement on the next line |
| 263 | `# if __name__ == '__main__':` | **Preserve** | Commented-out code |
| 264 | `#     config_parser = config_parser_t("v4r1.conf")` | **Preserve** | Commented-out code |
| 265 | `#     config_content = config_parser()` | **Preserve** | Commented-out code |
| 266 | `#     config_content.dump()` | **Preserve** | Commented-out code |

*Note:* Lines 87, 91, 92 contain `#` as string literals in code, not comments. Excluded from comment count.

**Actual non-license comments:** 9 (lines 77, 125, 137, 169, 218, 222, 263, 264, 265, 266 — but 263-266 are a block)

**Estimated % removable:** ~33% (3 of 9: lines 77, 218, 222)

**Preserved comments:** Lines 125, 137, 169, 263-266 (commented-out main block)

---

### 5. `instruction.py`

**Total non-license comments:** 1

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 35 | `# like macro_c_clear_t. this is a hack` | **Preserve** | HACK marker; documents a design workaround |

**Estimated % removable:** 0%

**Preserved comments:** Line 35

---

### 6. `macro.py`

**Total non-license comments:** 3

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 69 | `# 1st, overwrite original declared arguments` | **Preserve** | Documents the ordering invariant of a 3-step macro expansion protocol |
| 78 | `# 2nd, do the emit` | **Preserve** | Step 2 of the macro expansion protocol |
| 83 | `# last, restore arg to default value.` | **Preserve** | Step 3 of the macro expansion protocol |

*Note:* These three comments together document a critical ordering invariant (overwrite → emit → restore) in macro inline expansion. Removing any one breaks the logical grouping.

**Estimated % removable:** 0%

**Preserved comments:** Lines 69, 78, 83

---

### 7. `mbb.py`

**Total non-license comments:** 1

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 120 | `# note: copy to here!` | **Remove** | Restates the `list()` constructor call on the same line |

**Estimated % removable:** 100%

**Preserved comments:** None

---

### 8. `mc.py`

**Total non-license comments:** 10

| Line | Comment | Classification | Rationale / Condensed Version |
|------|---------|---------------|-------------------------------|
| 32 | `# NOTE: if following set to True, better parse '-V 0' to conv_driver` | **Preserve** | WARNING/NOTE: debug flags that break correctness |
| 33 | `# since result can never be correct` | **Preserve** | WARNING continuation: explains correctness impact |
| 149 | `# TODO: exception check` | **Preserve** | TODO marker |
| 163 | `# ignore_list.extend(['ds_write'])` | **Preserve** | Commented-out code (alternative debug filter) |
| 241 | `# manage the indent here` | **Remove** | Restates the assignment `self.indent = upper_emitter.indent` |
| 276 | `# for uniqueness` | **Preserve** | Explains the purpose of the `set()` (deduplication semantics) |
| 287 | `# TODO: better check valid emitter` | **Preserve** | TODO marker |
| 294 | `# Note! sort by name here!` | **Preserve** | Architectural note: ordering invariant for emit_all_unique |
| 351 | `#def _emit_unique_wrapper():` | **Preserve** | Commented-out code |
| 352 | `#    self.emit_unique(other)` | **Preserve** | Commented-out code (continuation) |
| 360 | `#other._emit_unique = _emit_unique_wrapper` | **Preserve** | Commented-out code |

**Estimated % removable:** ~10% (1 of 10: line 241)

**Preserved comments:** Lines 32, 33, 149, 163, 276, 287, 294, 351, 352, 360

---

### 9. `scheduler.py`

**Total non-license comments:** 40

This file is dominated by commented-out debug `print` statements and commented-out alternative algorithm implementations. All are classified as **Preserve** per the rules (commented-out code and debug/logging).

| Line | Comment | Classification | Rationale |
|------|---------|---------------|-----------|
| 130 | `#print(f"entering mbb merge start")` | **Preserve** | Commented-out debug logging |
| 134 | `#print(f"entering mbb merge middle")` | **Preserve** | Commented-out debug logging |
| 140 | `#print(f"entering mbb merge end")` | **Preserve** | Commented-out debug logging |
| 147 | `#print(f"entering mbb merge end")` | **Preserve** | Commented-out debug logging |
| 154 | `#print(f"entering mbb merge end")` | **Preserve** | Commented-out debug logging |
| 213 | `# mfma 32x32 inst allow at most 14` | **Preserve** | Hardware-specific performance assumption (MFMA instruction limit) |
| 217 | `# used in pattern_1` | **Preserve** | Documents which interleave pattern uses this parameter |
| 226 | `#assert interleave_space <= max_interleave_space, ...` | **Preserve** | Commented-out assertion (debug/safety check) |
| 232 | `# first check how many global load in mbb_1` | **Preserve** | Algorithm step documentation in complex interleaving logic |
| 233 | `# for x in mbb_1:` | **Preserve** | Commented-out code |
| 234 | `#     x.dump()` | **Preserve** | Commented-out debug code |
| 236 | `#assert mbb_have_global_mem(mbb_1[0])` | **Preserve** | Commented-out assertion |
| 244 | `#else:` | **Preserve** | Commented-out code |
| 245 | `#    break` | **Preserve** | Commented-out code |
| 246 | `#assert num_gmem != 0, ...` | **Preserve** | Commented-out assertion |
| 247 | `# assert num_v_c_clear in (0, 1)` | **Preserve** | Commented-out assertion |
| 250 | `# second decide how many global mem to interleave per interval` | **Preserve** | Algorithm step documentation |
| 251 | `# if num global mem bigger than this of mbb_0 length, need add more per interval` | **Preserve** | Explains the 2/3 ratio heuristic (performance assumption) |
| 253 | `#while num_gmem * gmem_per_interval >= ...` | **Preserve** | Commented-out alternative algorithm |
| 260 | `#mbb_1_left_per_interval = ...` | **Preserve** | Commented-out alternative formula |
| 262 | `#mbb_1_left_per_interval = num_mbb_1_left // num_mbb_0_left` | **Preserve** | Commented-out alternative formula |
| 263 | `#print(f"num_mbb_0_interleave_gmem:...")` | **Preserve** | Commented-out debug logging |
| 265 | `# finaly, go interleave` | **Preserve** | Algorithm step documentation |
| 274 | `# 0-emit gmem, v_clear, 1-other` | **Preserve** | Documents state machine enum values |
| 286 | `#print(f" ---- m0_idx:{m0_idx}, m1_idx:{m1_idx}")` | **Preserve** | Commented-out debug logging |
| 291 | `#self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1` | **Preserve** | Commented-out code (old approach) |
| 293 | `#print(f'      m1_idx:{m1_idx}')` | **Preserve** | Commented-out debug logging |
| 308 | `#self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1` | **Preserve** | Commented-out code (old approach) |
| 312 | `#m0_idx = 0` | **Preserve** | Commented-out code |
| 313 | `#m1_idx = 0` | **Preserve** | Commented-out code |
| 314 | `#for i in range(num_mbb_0_interleave_gmem):` | **Preserve** | Commented-out alternative algorithm block |
| 315 | `#    self._emit(self.call_mbb(mbb_0[m0_idx])) ; m0_idx += 1` | **Preserve** | Commented-out alternative algorithm block |
| 316 | `#    for j in range(gmem_per_interval):` | **Preserve** | Commented-out alternative algorithm block |
| 317 | `#        if m1_idx < num_gmem:` | **Preserve** | Commented-out alternative algorithm block |
| 318 | `#            self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1` | **Preserve** | Commented-out alternative algorithm block |
| 320 | `#for i in range(num_mbb_0_left):` | **Preserve** | Commented-out alternative algorithm block |
| 321 | `#    self._emit(self.call_mbb(mbb_0[m0_idx])) ; m0_idx += 1` | **Preserve** | Commented-out alternative algorithm block |
| 322 | `#    for j in range(mbb_1_left_per_interval):` | **Preserve** | Commented-out alternative algorithm block |
| 323 | `#        if m1_idx < len(mbb_1):` | **Preserve** | Commented-out alternative algorithm block |
| 324 | `#            self._emit(self.call_mbb(mbb_1[m1_idx])) ; m1_idx += 1` | **Preserve** | Commented-out alternative algorithm block |
| 325 | `# assert m0_idx == len(mbb_0)` | **Preserve** | Commented-out assertion |
| 351 | `#smem_per_interleave = (len(mbb_0) - 1 + num_smem - 1) // num_smem` | **Preserve** | Commented-out alternative formula |
| 353 | `#print(f'__ len(mbb_0):...")` | **Preserve** | Commented-out debug logging |
| 362 | `# print(f' --- inst:...")` | **Preserve** | Commented-out debug logging |
| 381 | `# TODO might have other type of scheduler` | **Preserve** | TODO marker |

**Estimated % removable:** 0%

**Preserved comments:** All 40 comments (lines 130, 134, 140, 147, 154, 213, 217, 226, 232, 233, 234, 236, 244, 245, 246, 247, 250, 251, 253, 260, 262, 263, 265, 274, 286, 291, 293, 308, 312-318, 320-325, 325, 351, 353, 362, 381)

---

### 10. `symbol.py`

**Total non-license comments:** 0

No comments beyond the license header. File contains only a class definition.

**Estimated % removable:** N/A

**Preserved comments:** None

---

## Cross-File Observations

1. **`scheduler.py` is the heaviest comment file** (40 comments, all preserved) — dominated by commented-out debug `print` statements and alternative algorithm implementations that were iteratively developed. These represent the algorithmic history of the interleaving scheduler.

2. **Hardware-specific comments cluster in `amdgpu.py`** — 8 comments document gfx-arch-specific assembler behaviors (gfx1250, gfx1170+, gfx1030, etc.), SGPR/VGPR counting rules, and code object format details. These are irreplaceable hardware documentation.

3. **Debug flags in `mc.py` (lines 32-33)** carry a critical WARNING: enabling them produces incorrect results. This is a safety note that must be preserved.

4. **Only ~11% of comments are removable** — the codebase is already lean on comments. The removable ones are primarily obvious restatements (`# in byte`, `# for multiple files`, `# default set to 8`, `# hard code to 0`).

5. **No Compress candidates were found** — no comments are excessively long or repetitive enough to warrant condensing rather than remove/preserve.

6. **Commented-out code is pervasive** in `scheduler.py` and `compile.py` — these serve as algorithmic alternatives and debug scaffolding. Per the classification rules, all are preserved.

---

## Complete List of Removable Comments

| File | Line | Comment | Reason |
|------|------|---------|--------|
| `amdgpu.py` | 224 | `# in byte` | Obvious from `self.lds_size` variable name |
| `amdgpu.py` | 592 | `# default set to 8` | Self-documenting `8` literal |
| `amdgpu.py` | 595 | `# hard code to 0` | Restates `0` literal on same line |
| `compile.py` | 200 | `# for multiple files` | Obvious from `is list` type check |
| `compile.py` | 226 | `# for multiple files` | Obvious from `is list` type check |
| `config_parser.py` | 77 | `# return a list of section, each section is key-value pair` | Restates function purpose |
| `config_parser.py` | 218 | `# safe to recursively call here, to better create a value type` | Restates visible recursive call |
| `config_parser.py` | 222 | `# finaly return string` | Restates `return value` on next line |
| `mbb.py` | 120 | `# note: copy to here!` | Restates `list()` constructor |
| `mc.py` | 241 | `# manage the indent here` | Restates `self.indent = ...` assignment |

---

## Complete List of Compress Candidates

None identified. No comments in the codebase are excessively long or repetitive enough to warrant condensing.

---

*Report generated from read-only analysis of `python/codegen/` directory.*
