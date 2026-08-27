#!/usr/bin/env python3
"""
Assembles one comprehensive "master" .config file per (direction, precision) for gfx1250,
combining every validated tunable section from this project's many narrow, single-mechanism
config files (config/igemm_{fwd,bwd,wrw}_gtc_gfx1250_nhwc_{precision}_*.config) into one file
-- mirroring gfx950/gfx942's existing structure (one ~100-200-tunable file per direction/
precision, see script/gen_gfx950_conv_split_kernel.sh), so a single `conv_driver.exe` build
searches the FULL candidate set and reports the true fastest kernel per shape, matching the
gfx950 user experience.

This is a pure text-level union: each source .config file's `[igemm_{direction}_gtc]`
sections are extracted verbatim and deduplicated (by normalized content -- so a tunable that
happens to appear identically in two source files, e.g. via legacy copy-paste, is only
included once). No tunable values are invented or modified. Kernel-name collisions (which
would previously have been silent/undetected, since several tail-relief tunables were not
folded into the kernel name until this change) are surfaced as a real igemm_codegen.py
build failure -- see docs/gfx1250_wmma_layout.md's master-config phase for the name-folding
fix this relies on.

Usage:
    python3 script/build_gfx1250_master_configs.py [--write]

Without --write, only reports what WOULD be written (dry run). With --write, generates
config/igemm_{direction}_gtc_gfx1250_nhwc_{precision}_all.config for every (direction,
precision) pair found.
"""
import argparse
import glob
import os
import re
import sys
from collections import OrderedDict

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')

FILENAME_RE = re.compile(r'^igemm_(fwd|bwd|wrw)_gtc_gfx1250_nhwc_(fp16|bf16|fp32|int8)(_.*)?\.config$')

SECTION_RE = re.compile(r'^\[([a-zA-Z0-9_]+)\]\s*$')


def parse_config_file(path):
    """Returns (codegen_header_lines, [(section_name, section_body_lines), ...])."""
    with open(path) as f:
        lines = f.readlines()

    codegen_lines = []
    sections = []
    cur_name = None
    cur_body = []
    cur_comment_block = []
    in_codegen = False

    for line in lines:
        m = SECTION_RE.match(line.strip())
        if m:
            if cur_name is not None:
                sections.append((cur_name, cur_comment_block + cur_body))
            name = m.group(1)
            if name == 'codegen':
                in_codegen = True
                cur_name = None
                cur_body = []
                cur_comment_block = []
                continue
            else:
                in_codegen = False
                cur_name = name
                cur_body = [line]
                cur_comment_block = []
                continue
        if in_codegen:
            codegen_lines.append(line)
        elif cur_name is not None:
            cur_body.append(line)
        else:
            # comment/banner lines before the first section (or between sections) --
            # attach to whichever section follows.
            cur_comment_block.append(line)
    if cur_name is not None:
        sections.append((cur_name, cur_comment_block + cur_body))
    return codegen_lines, sections


def normalize(body_lines):
    """Normalize a section body for dedup comparison -- strip pure-comment/blank lines and
    surrounding whitespace, keep only the actual key=value content lines."""
    keep = []
    for l in body_lines:
        s = l.strip()
        if not s or s.startswith('#'):
            continue
        keep.append(s)
    return '\n'.join(keep)


# conv_driver.cpp computes is_wmma_f16_acc/is_wmma_bf16_acc/is_wmma_atomic_pack_bf16 (and the
# derived dtype_alloc_byte output-buffer width) ONCE from tunables[0], not per-tunable inside
# the search loop -- confirmed by reading conv_driver.cpp directly (lines ~669-780) after this
# script's first attempt at a master file produced a REAL, reproducible valid:n for wmma_acc_f16/
# bf16 kernels that pass fine standalone. Mixing an accumulate-width-variant section into a file
# whose first tunable doesn't share that width silently corrupts verification for it -- a
# pre-existing driver limitation this consolidation is the first thing to expose (gfx950/942
# never had a per-tunable output-width concept at all). Excluded here rather than fixed in
# conv_driver.cpp (which would need buffer allocation restructured to happen per-tunable, not
# once upfront) -- these keep working fine as their own separate, narrower config files.
ACCUMULATE_WIDTH_KEYS = ('wmma_acc_f16', 'wmma_acc_bf16', 'atomic_pack_bf16')


def has_accumulate_width_variant(body_lines):
    for l in body_lines:
        s = l.strip()
        if s.startswith('#'):
            continue
        for key in ACCUMULATE_WIDTH_KEYS:
            if re.match(rf'^{key}\s*=\s*1\b', s):
                return key
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='actually write the _all.config files (default: dry run)')
    args = ap.parse_args()

    groups = OrderedDict()  # (direction, precision) -> list of file paths
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, 'igemm_*_gtc_gfx1250_nhwc_*.config'))):
        fname = os.path.basename(path)
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        direction, precision = m.group(1), m.group(2)
        if fname.endswith('_all.config'):
            continue  # skip any previously-generated master file itself
        groups.setdefault((direction, precision), []).append(path)

    for (direction, precision), paths in groups.items():
        codegen_lines = None
        seen = OrderedDict()  # normalized body -> (source_file, section_name, body_lines)
        dup_count = 0
        excluded = []  # (source_file, reason)
        for path in paths:
            cg, sections = parse_config_file(path)
            if codegen_lines is None:
                codegen_lines = cg
            for name, body in sections:
                key = normalize(body)
                if not key:
                    continue
                variant = has_accumulate_width_variant(body)
                if variant is not None:
                    excluded.append((os.path.basename(path), variant))
                    continue
                if key in seen:
                    dup_count += 1
                    continue
                seen[key] = (os.path.basename(path), name, body)

        out_name = f"igemm_{direction}_gtc_gfx1250_nhwc_{precision}_all.config"
        out_path = os.path.join(CONFIG_DIR, out_name)
        n_sections = len(seen)
        excl_note = f", {len(excluded)} accumulate-width-variant sections excluded ({', '.join(sorted(set(v for _, v in excluded)))})" if excluded else ""
        print(f"{direction}/{precision}: {len(paths)} source files -> {n_sections} unique tunable sections "
              f"({dup_count} exact duplicates skipped{excl_note}) -> {out_name}", file=sys.stderr)

        if args.write:
            with open(out_path, 'w') as f:
                f.write("[codegen]\n")
                f.writelines(codegen_lines)
                f.write("\n")
                f.write(f"#########################################################################################\n")
                f.write(f"# Master config: every validated {direction}/{precision} gfx1250 tunable combination from\n")
                f.write(f"# this project's individual mechanism config files, unioned into one file so a single\n")
                f.write(f"# conv_driver.exe build searches the FULL candidate set and reports the true fastest\n")
                f.write(f"# kernel per shape -- mirroring gfx950/942's single comprehensive per-direction/precision\n")
                f.write(f"# config file (see script/gen_gfx950_conv_split_kernel.sh). Generated by\n")
                f.write(f"# script/build_gfx1250_master_configs.py from {len(paths)} source files; re-run that\n")
                f.write(f"# script after adding a new gfx1250 config file to pick it up here too.\n")
                if excluded:
                    f.write(f"#\n")
                    f.write(f"# NOT included: {len(excluded)} accumulate-width-variant section(s) "
                            f"({', '.join(sorted(set(v for _, v in excluded)))}) from "
                            f"{', '.join(sorted(set(s for s, _ in excluded)))} -- conv_driver.cpp computes\n")
                    f.write(f"# is_wmma_f16_acc/is_wmma_bf16_acc/is_wmma_atomic_pack_bf16 ONCE from tunables[0],\n")
                    f.write(f"# not per-tunable, so mixing a different output-buffer-width kernel into this file\n")
                    f.write(f"# would silently corrupt verification for it (confirmed: passes standalone, fails\n")
                    f.write(f"# here). These remain usable via their own individual config files.\n")
                f.write(f"#########################################################################################\n")
                for key, (src, name, body) in seen.items():
                    f.write(f"\n# --- from {src} ---\n")
                    f.writelines(body)
            print(f"  wrote {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
