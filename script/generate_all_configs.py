#!/usr/bin/env python3
"""
Generate comprehensive _all.config files per tile shape with every valid tunable
combination for gfx1250 WMMA.

Each output file covers ONE tile shape for ONE (direction, precision) pair,
keeping each built .hsaco under the assembler's branch-range limit (±32KB
s_cbranch/s_branch immediate — the old monolithic _all.config with hundreds of
sections hit `branch size exceeds simm16`).

Both benchmark scripts are updated to try all tile-shape _all.config files
per (direction, precision) → conv_driver.exe searches each file's candidates
independently and the script reports the overall fastest.

Usage:
    python3 script/generate_all_configs.py [--write]
"""
import argparse
import os
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, 'config')

# (direction, precision, tile_m, tile_n, gemm_k, source_config)
BASE_SECTIONS = [
    # fwd
    ('fwd', 'fp16', 128, 128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config'),
    ('fwd', 'fp16', 64,  64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config'),
    ('fwd', 'fp16', 128, 64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16_128x64.config'),
    ('fwd', 'fp16', 64,  128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_fp16_64x128.config'),
    ('fwd', 'bf16', 128, 128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config'),
    ('fwd', 'bf16', 64,  64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16.config'),
    ('fwd', 'bf16', 128, 64,  32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_128x64.config'),
    ('fwd', 'bf16', 64,  128, 32, 'config/igemm_fwd_gtc_gfx1250_nhwc_bf16_64x128.config'),
    ('fwd', 'fp32', 128, 128, 4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32.config'),
    ('fwd', 'fp32', 64,  64,  4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32.config'),
    ('fwd', 'fp32', 128, 64,  4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32_128x64.config'),
    ('fwd', 'fp32', 64,  128, 4,  'config/igemm_fwd_gtc_gfx1250_nhwc_fp32_64x128.config'),
    # bwd
    ('bwd', 'fp16', 128, 128, 32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16.config'),
    ('bwd', 'fp16', 64,  64,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16.config'),
    ('bwd', 'fp16', 32,  32,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_fp16_32x32.config'),
    ('bwd', 'bf16', 128, 128, 32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16.config'),
    ('bwd', 'bf16', 64,  64,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16.config'),
    ('bwd', 'bf16', 32,  32,  32, 'config/igemm_bwd_gtc_gfx1250_nhwc_bf16_32x32.config'),
    ('bwd', 'fp32', 128, 128, 4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config'),
    ('bwd', 'fp32', 64,  64,  4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32.config'),
    ('bwd', 'fp32', 32,  32,  4,  'config/igemm_bwd_gtc_gfx1250_nhwc_fp32_32x32.config'),
    # wrw
    ('wrw', 'fp16', 128, 128, 32, 'config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config'),
    ('wrw', 'fp16', 64,  64,  32, 'config/igemm_wrw_gtc_gfx1250_nhwc_fp16.config'),
    ('wrw', 'bf16', 128, 128, 32, 'config/igemm_wrw_gtc_gfx1250_nhwc_bf16.config'),
    ('wrw', 'bf16', 64,  64,  32, 'config/igemm_wrw_gtc_gfx1250_nhwc_bf16.config'),
    ('wrw', 'fp32', 128, 128, 4,  'config/igemm_wrw_gtc_gfx1250_nhwc_fp32.config'),
    ('wrw', 'fp32', 64,  64,  4,  'config/igemm_wrw_gtc_gfx1250_nhwc_fp32.config'),
]

# Binary tunables to toggle combinatorially
# Phase 67: added saddr_global_load -- was previously only ever tested in its own
# bespoke single-feature _saddr.config, never combined with direct_store/tail-relief/
# lds_double_buffer/wmma_setprio/local_prefetch_num/epilogue_lds_pad in the searched
# corpus. ds_load_tr_b is NOT added here: it was promoted to an unconditional default
# for bwd/wrw fp16/bf16 (igemm_base.py), so every combination below already gets it
# for free without needing a combinatorial toggle.
FLAGS = [
    'direct_store', 'gemm_k_global_split', 'wmma_m_tail', 'wmma_n_tail',
    'tdm_global_load', 'lds_double_buffer', 'wmma_setprio',
    'local_prefetch_num', 'main_loop_interleave', 'epilogue_lds_pad',
    'saddr_global_load',
]


def parse_sections(path):
    """Return [(section_name, [lines]), ...]."""
    with open(os.path.join(REPO_ROOT, path)) as f:
        text = f.read()
    sections = []
    for m in re.finditer(r'^\[(igemm_\w+)\]\s*\n(.*?)(?=^\[igemm_\w+\]|\Z)', text,
                         re.MULTILINE | re.DOTALL):
        sections.append((m.group(1), m.group(2).strip().split('\n')))
    return sections


def find_base(sections, direction, tile_m):
    """Find the section body with gemm_m_per_block == tile_m."""
    for name, body in sections:
        for line in body:
            m = re.match(r'gemm_m_per_block\s*=\s*(\d+)', line.strip())
            if m and int(m.group(1)) == tile_m:
                return name, [l + '\n' for l in body if l.strip()]
    return None, None


def is_valid(direction, precision, tile_m, tile_n, gemm_k, vals):
    gs = vals.get('gemm_k_global_split', 0)
    ds = vals.get('direct_store', 0)
    mt = vals.get('wmma_m_tail', 0)
    nt = vals.get('wmma_n_tail', 0)
    tdm = vals.get('tdm_global_load', 0)
    ldb = vals.get('lds_double_buffer', 0)
    sp  = vals.get('wmma_setprio', 0)
    lpn = vals.get('local_prefetch_num', 1)
    mli = vals.get('main_loop_interleave', 0)
    elp = vals.get('epilogue_lds_pad', 0)
    sa  = vals.get('saddr_global_load', 0)
    ik  = 4 if precision == 'fp32' else 32

    # saddr_global_load exclusions mirror the exact hard asserts in
    # igemm_{fwd,bwd,wrw}_gtc_wmma_nhwc.py (tdm/interleave/gsplit all explicitly
    # asserted incompatible; row_repeat_a/b>1 -- i.e. the asymmetric 128x64/64x128
    # tile shapes -- also asserted incompatible, proxied here via tile_m != tile_n
    # the same way the nt/tdm/mli rules above already do).
    if sa and (tdm or mli or gs): return False
    if sa and tile_m != tile_n: return False
    # Phase 67: fwd's saddr_global_load + wmma_n_tail is a REAL, newly-discovered
    # correctness bug -- confirmed via hardware A/B (plain wmma_n_tail alone passes
    # `valid:y` on an exact-fit shape; the identical shape with saddr_global_load
    # ALSO set fails `valid:n`; saddr_global_load + wmma_m_tail alone is fine). Not
    # root-caused (likely fwd's B-operand N-boundary address computation not
    # accounting for saddr's different addressing path) -- excluded here rather than
    # silently shipped broken in the searched corpus. bwd's identical combination
    # (saddr_global_load + wmma_n_tail) hardware-validated fine and is NOT excluded --
    # this is fwd-specific. See docs/gfx1250_optimization_backlog.md.
    if sa and nt and direction == 'fwd': return False

    if ds and gs:            return False
    if gs and direction != 'wrw' and (mt or nt): return False
    if gs and elp:           return False
    if nt and tile_m != tile_n: return False  # wmma_n_tail requires row_repeat_b==1
    if tdm and tile_m != tile_n: return False  # TDM not supported with row_repeat_a>1
    # NOTE: VGPR-budget exclusions removed -- now handled by a post-build filter
    if tdm and gs:           return False
    if tdm and (lpn > 1 or mli): return False
    if tdm and tile_m != 128: return False
    if tdm and direction != 'fwd' and (mt or nt): return False
    if mli and lpn > 1:      return False
    if mli and not ldb:      return False
    if mli and gs:           return False
    if mli and tile_m != tile_n: return False
    if lpn > 1 and precision in ('fp16', 'bf16') and tile_m == 128: return False
    if elp and tile_m == 128: return False
    if elp and ds:           return False
    if direction == 'wrw' and (mt or nt) and not gs: return False
    if lpn > 1 and gemm_k <= ik: return False
    if mli and gemm_k <= ik: return False
    return True


def gen_combos():
    """Yield all (dir, prec, tm, tn, gk, src, vals_dict)."""
    from itertools import product
    for direction, precision, tile_m, tile_n, gemm_k, src in BASE_SECTIONS:
        for bits in product([0, 1], repeat=len(FLAGS)):
            vals = {FLAGS[i]: bits[i] for i in range(len(FLAGS))}
            # local_prefetch_num: bit 0 -> 1, bit 1 -> 2
            vals['local_prefetch_num'] = 2 if vals['local_prefetch_num'] == 1 else 1
            if is_valid(direction, precision, tile_m, tile_n, gemm_k, vals):
                yield direction, precision, tile_m, tile_n, gemm_k, vals


def section_key(body_lines):
    """Deterministic dedup key: all non-comment tunable lines, sorted."""
    tunable_lines = []
    for l in body_lines:
        s = l.strip()
        if not s or s.startswith('#') or s.startswith(';'):
            continue
        tunable_lines.append(s)
    return '\n'.join(sorted(tunable_lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    # Pre-parse all source configs
    src_cache = {}
    for entry in BASE_SECTIONS:
        _, _, _, _, _, src = entry
        if src not in src_cache:
            src_cache[src] = parse_sections(src)

    base_cache = {}
    for direction, precision, tile_m, tile_n, gemm_k, src in BASE_SECTIONS:
        key = (direction, precision, tile_m, tile_n, gemm_k)
        if key in base_cache:
            continue
        sname, sbody = find_base(src_cache[src], direction, tile_m)
        if sbody is None:
            print(f"WARNING: no {tile_m}x{tile_n} section in {src}", file=sys.stderr)
            continue
        base_cache[key] = sbody

    # Generate all combinations
    combos = list(gen_combos())
    print(f"Total valid combinations: {len(combos)}")

    # Group by (direction, precision, tile_m, tile_n, gemm_k)
    per_tile = defaultdict(list)
    for direction, precision, tile_m, tile_n, gemm_k, vals in combos:
        per_tile[(direction, precision, tile_m, tile_n, gemm_k)].append(vals)

    total_files = 0
    total_sections = 0
    for (direction, precision, tile_m, tile_n, gemm_k), combos_list in sorted(per_tile.items()):
        # Output file name: igemm_{dir}_gtc_gfx1250_nhwc_{prec}_{tm}x{tn}_all.config
        out_name = f'igemm_{direction}_gtc_gfx1250_nhwc_{precision}_{tile_m}x{tile_n}_all.config'
        out_path = os.path.join(CONFIG_DIR, out_name)

        # Read existing file header if present
        codegen_header = [
            '[codegen]\n', "arch = 'gfx1250'\n", "code_object = 'cov3'\n", "mode = 'flat'\n"
        ]

        base_body = base_cache.get((direction, precision, tile_m, tile_n, gemm_k))
        if base_body is None:
            continue

        out_lines = list(codegen_header)
        out_lines.append('\n')
        out_lines.append(f"{'#' * 89}\n")
        out_lines.append(f"# Master config: all valid tunable combos for {direction}/{precision} "
                         f"{tile_m}x{tile_n}x{gemm_k}\n")
        out_lines.append(f"# Generated by script/generate_all_configs.py\n")
        out_lines.append(f"# {len(combos_list)} combinatorial variants\n")
        out_lines.append(f"{'#' * 89}\n\n")

        seen = set()
        for vals in sorted(combos_list, key=lambda v: sorted(v.items())):
            # Clone base body and append non-default tunable flags
            new_body = list(base_body)

            # Gather non-default tunable lines
            extra = []
            for flag in FLAGS:
                val = vals.get(flag)
                if val is not None and val != 0 and val != 'SCOPE_SYS':
                    extra.append(f"{flag:25s} = {val}\n")

            if extra:
                # Insert after last non-comment body line
                insert_at = len(new_body)
                for i in range(len(new_body) - 1, -1, -1):
                    s = new_body[i].strip()
                    if s and not s.startswith('#') and not s.startswith(';'):
                        insert_at = i + 1
                        break
                for i, el in enumerate(extra):
                    new_body.insert(insert_at + i, el)

            key = section_key(new_body)
            if key in seen:
                continue
            seen.add(key)

            # Build label
            active = [k for k, v in sorted(vals.items()) if v not in (0, 'SCOPE_SYS')]
            label = '+'.join(active) if active else 'base'

            section_name = f'igemm_{direction}_gtc'
            out_lines.append(f"# --- {tile_m}x{tile_n}x{gemm_k} +{label} ---\n")
            out_lines.append(f"[{section_name}]\n")
            out_lines.extend(new_body)
            out_lines.append('\n')

        total_files += 1
        total_sections += len(seen)
        print(f"{out_name}: {len(seen)} sections ({len(out_lines)} lines)")

        if args.write:
            with open(out_path, 'w') as f:
                f.writelines(out_lines)

    print(f"\n{total_files} files, {total_sections} total sections")

    # Also write a combined _all.config with the original naming for backward compat
    # with the diverse benchmark script — it expects one file per (dir, prec)
    # named igemm_{dir}_gtc_gfx1250_nhwc_{prec}_all.config
    print("\n--- backward-compat combined files ---")
    combined = defaultdict(list)
    for (direction, precision, tile_m, tile_n, gemm_k), combos_list in sorted(per_tile.items()):
        combined[(direction, precision)].append((tile_m, tile_n, gemm_k, combos_list))

    for (direction, precision), tile_list in sorted(combined.items()):
        out_name = f'igemm_{direction}_gtc_gfx1250_nhwc_{precision}_all.config'
        out_path = os.path.join(CONFIG_DIR, out_name)
        out_lines = list(codegen_header)
        out_lines.append('\n')
        out_lines.append(f"{'#' * 89}\n")
        out_lines.append(f"# Combined master config for {direction}/{precision}\n")
        out_lines.append(f"# Generated by script/generate_all_configs.py\n")
        out_lines.append(f"# Per-tile-shape variants also available as:\n")
        for tm, tn, gk, _ in tile_list:
            out_lines.append(f"#   igemm_{direction}_gtc_gfx1250_nhwc_{precision}_{tm}x{tn}_all.config\n")
        out_lines.append(f"{'#' * 89}\n\n")

        total = 0
        for tile_m, tile_n, gemm_k, combos_list in tile_list:
            base_body = base_cache.get((direction, precision, tile_m, tile_n, gemm_k))
            if base_body is None:
                continue
            seen = set()
            for vals in sorted(combos_list, key=lambda v: sorted(v.items())):
                new_body = list(base_body)
                extra = []
                for flag in FLAGS:
                    val = vals.get(flag)
                    if val is not None and val != 0 and val != 'SCOPE_SYS':
                        extra.append(f"{flag:25s} = {val}\n")
                if extra:
                    insert_at = len(new_body)
                    for i in range(len(new_body) - 1, -1, -1):
                        s = new_body[i].strip()
                        if s and not s.startswith('#') and not s.startswith(';'):
                            insert_at = i + 1
                            break
                    for i, el in enumerate(extra):
                        new_body.insert(insert_at + i, el)
                key = section_key(new_body)
                if key in seen:
                    continue
                seen.add(key)
                active = [k for k, v in sorted(vals.items()) if v not in (0, 'SCOPE_SYS')]
                label = '+'.join(active) if active else 'base'
                section_name = f'igemm_{direction}_gtc'
                out_lines.append(f"# --- {tile_m}x{tile_n}x{gemm_k} +{label} ---\n")
                out_lines.append(f"[{section_name}]\n")
                out_lines.extend(new_body)
                out_lines.append('\n')
                total += 1

        print(f"{out_name}: {total} sections ({len(out_lines)} lines)")
        if args.write:
            with open(out_path, 'w') as f:
                f.writelines(out_lines)

    if not args.write:
        print("\nDry run -- use --write to actually generate files.")


if __name__ == '__main__':
    main()