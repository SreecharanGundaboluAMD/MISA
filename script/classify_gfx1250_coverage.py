#!/usr/bin/env python3
"""
Classifies a large corpus of real MIOpenDriver conv shapes (shapes where an existing
MISA-authored solver -- ConvAsmImplicitGemmGTCDynamic{Fwd,Bwd,Wrw}XdlopsNHWC -- won
MIOpen's solver search, i.e. real, validated-fast shapes on some other architecture)
against what MISA's gfx1250 WMMA backend can build TODAY, and what it would take to
close each gap. Pure static analysis (GEMM-dimension arithmetic + known mechanism
support, no hardware runs) -- the corpus is far too large (~95k shapes) to run on
real hardware one at a time, and the question here is coverage, not per-shape timing.

Usage:
    python3 script/classify_gfx1250_coverage.py \
        convasmimplicitgemmgtcdynamic_fwd.txt \
        convasmimplicitgemmgtcdynamic_bwd.txt \
        convasmimplicitgemmgtcdynamic_wrw.txt \
        [--csv-out FILE] [--md-out FILE]

Each input file is expected in the format produced by a MIOpen solver-search trace:
    # matched solver: ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC  time=0.0075ms
    MIOpenDriver convbfp16 -n 1 -c 1 -k 64 -H 1 -W 1 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -b 0 -F 1 -t 1
The direction is inferred from the filename (fwd/bwd/wrw), not the -F flag (some
corpora only vary -F for other reasons) -- pass the right file for the right direction.
"""
import argparse
import re
import sys
from collections import defaultdict, Counter

MIOPEN_PRECISION_MAP = {
    'conv': 'fp32',
    'convfp16': 'fp16',
    'convbfp16': 'bf16',
    'convint8': 'int8',
    'convint4': 'int4',
}

# gemm_k_per_block for the exact-fit (no K-tail) base case, per precision -- fp16/bf16
# share the 32-element WMMA K-instruction width, int8 packs 2x (64), fp32 needs 4
# (matches inst_wmma.k for that precision). int4 has no gfx1250 WMMA support at all.
BASE_GEMM_K_PER_BLOCK = {'fp16': 32, 'bf16': 32, 'fp32': 4}

DRIVER_LINE_RE = re.compile(r'^MIOpenDriver\s+(\S+)\s+(.*)$')
SOLVER_LINE_RE = re.compile(r'#\s*matched solver:\s*(\S+)\s+time=([\d.]+)ms')


def parse_args_str(rest):
    toks = rest.split()
    args = {}
    layout = {'in_layout': 'NCHW', 'fil_layout': 'NCHW', 'out_layout': 'NCHW'}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ('--in_layout', '--fil_layout', '--out_layout'):
            layout[t[2:]] = toks[i + 1]
            i += 2
        elif t.startswith('--'):
            # unrecognized long flag -- skip its value defensively
            i += 2
        elif t.startswith('-') and len(t) > 1 and not t[1].isdigit():
            key = t[1:]
            if i + 1 < len(toks):
                args[key] = toks[i + 1]
            i += 2
        else:
            i += 1
    return args, layout


def conv_out_size(in_size, pad, dilation, ksize, stride):
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) // stride + 1


def parse_file(path, direction):
    """Yields dicts, one per shape entry."""
    pending_solver = None
    pending_time = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            m = SOLVER_LINE_RE.search(line)
            if m:
                pending_solver, pending_time = m.group(1), float(m.group(2))
                continue
            m = DRIVER_LINE_RE.match(line)
            if not m:
                continue
            mode, rest = m.group(1), m.group(2)
            precision = MIOPEN_PRECISION_MAP.get(mode, 'unknown:' + mode)
            args, layout = parse_args_str(rest)
            try:
                shape = {
                    'direction': direction,
                    'precision': precision,
                    'n': int(args.get('n', 1)),
                    'c': int(args.get('c', 1)),
                    'k': int(args.get('k', 1)),
                    'H': int(args.get('H', 1)),
                    'W': int(args.get('W', 1)),
                    'y': int(args.get('y', 1)),
                    'x': int(args.get('x', 1)),
                    'p': int(args.get('p', 0)),
                    'q': int(args.get('q', 0)),
                    'u': int(args.get('u', 1)),
                    'v': int(args.get('v', 1)),
                    'l': int(args.get('l', 1)),
                    'j': int(args.get('j', 1)),
                    'g': int(args.get('g', 1)),
                    'in_layout': layout['in_layout'],
                    'fil_layout': layout['fil_layout'],
                    'out_layout': layout['out_layout'],
                    'solver': pending_solver,
                    'ref_time_ms': pending_time,
                }
            except (ValueError, KeyError):
                continue
            yield shape
            pending_solver, pending_time = None, None


def compute_gemm_dims(shape):
    d = shape
    ho = conv_out_size(d['H'], d['p'], d['l'], d['y'], d['u'])
    wo = conv_out_size(d['W'], d['q'], d['j'], d['x'], d['v'])
    g = d['g']
    if d['direction'] == 'fwd':
        gm, gn, gk = d['n'] * ho * wo, d['k'] // g, d['c'] // g
    elif d['direction'] == 'bwd':
        gm, gn, gk = d['n'] * d['H'] * d['W'], d['c'] // g, d['k'] // g
    else:  # wrw
        gm, gn, gk = d['k'] // g, d['c'] // g, d['n'] * ho * wo
    return ho, wo, gm, gn, gk


def fits_tile(dim):
    return dim % 128 == 0 or dim % 64 == 0


def classify(shape, assume_nhwc=False):
    """Returns (category, note) -- category is a short machine-readable tag, note is
    a one-line human-readable reason, used both for aggregation and for picking
    representative examples.

    assume_nhwc: when True, skips the layout check entirely (models the hypothetical
    "every shape gets transposed to NHWC before dispatch, by MIOpen or otherwise" --
    NOT something MISA itself does today, see docs/gfx1250_wmma_layout.md's decision
    to leave layout conversion to the caller) so the remaining classification reflects
    pure GEMM-shape/mechanism coverage on the NHWC-only subset.
    """
    d = shape

    if not assume_nhwc and (d['in_layout'] != 'NHWC' or d['fil_layout'] != 'NHWC' or d['out_layout'] != 'NHWC'):
        return 'gap_layout_not_nhwc', (
            f"layout={d['in_layout']}/{d['fil_layout']}/{d['out_layout']} -- gfx1250 WMMA "
            f"kernels are NHWC-only; MISA deliberately does not transpose NCHW<->NHWC itself "
            f"(left to the caller, e.g. MIOpen's own solver-wrapping) -- see "
            f"docs/gfx1250_wmma_layout.md")

    if d['precision'] not in BASE_GEMM_K_PER_BLOCK:
        return 'gap_precision_unsupported', f"precision={d['precision']} has no gfx1250 WMMA support at all"

    if d['c'] % d['g'] != 0 or d['k'] % d['g'] != 0:
        return 'gap_invalid_group', f"c={d['c']} or k={d['k']} not divisible by g={d['g']} -- malformed shape"

    if d['g'] == d['c'] and d['g'] > 1:
        # NOTE: g>1 is load-bearing here, not defensive. g==c==1 (a single-group,
        # single-input-channel conv) technically satisfies the textbook "group_count ==
        # in_channels" depthwise definition too, but with only ONE group it degenerates
        # to an entirely ordinary conv -- MISA's group handling is generic (group_idx=0
        # trivially), there's nothing depthwise-specific about it, and it's structurally
        # identical to any other real conv with a small channel count. The actual
        # architectural pain (many tiny independent per-group GEMMs, hence a dedicated
        # Toeplitz-style kernel elsewhere, not an igemm approach) only shows up once g is
        # actually large. Verified against this corpus: every g==c shape here has g==1
        # (checked directly, zero g>1 depthwise shapes exist in this corpus) -- an earlier
        # version of this check flagged all of them as "depthwise" purely from g==c,
        # which was wrong for every single one of them.
        return 'gap_depthwise', (
            "g==c>1 (depthwise) -- architecturally out of scope for MISA's igemm approach "
            "on every architecture, not gfx1250-specific; a degenerate 1-wide GEMM per "
            "group has no efficient WMMA tile shape")

    ho, wo, gm, gn, gk = compute_gemm_dims(d)
    if ho <= 0 or wo <= 0:
        return 'degenerate_zero_output', (
            f"computed ho={ho} wo={wo} -- filter doesn't fit the input even once "
            f"(H={d['H']},W={d['W']},y={d['y']},x={d['x']},p={d['p']},q={d['q']} with no valid "
            f"conv window) -- not a real conv, likely a MIOpen solver-robustness edge case")

    prec = d['precision']
    base_k = BASE_GEMM_K_PER_BLOCK[prec]
    m_fits, n_fits, k_fits = fits_tile(gm), fits_tile(gn), (gk % base_k == 0)
    needs = set()
    if not m_fits:
        needs.add('M')
    if not n_fits:
        needs.add('N')
    if not k_fits:
        needs.add('K')

    if not needs:
        return 'supported_exact_fit', f"gm={gm},gn={gn},gk={gk} all fit a 128 or 64 tile exactly"

    # fwd/bwd's non-atomic epilogue (coalescing_store_wmma.py's LDS-reshuffle path) stores
    # in vectorized 4-wide chunks (vector_write_out=4) -- when wmma_n_tail is active (i.e.
    # 'N' in needs here, since every _mtail/_ntail/_mntail/_tdm/_ktail config that covers an
    # N-needing shape necessarily sets wmma_n_tail=1), the EXEC-mask guard only checks the
    # group's FIRST column, so a 4-column group straddling a non-multiple-of-4 gemm_n writes
    # the out-of-range tail columns too -- confirmed unbuildable on real hardware
    # (fwd fp32 n=1,c=1,k=1,H=W=1760 and n=256,c=3,k=3,H=W=32, both gemm_n=k in {1,3}, every
    # kernel reports "not applicable"). wrw's atomic epilogue is scalar-per-element (no
    # vectorized grouping at all), so this constraint does NOT apply there. Previously
    # unmodeled here -- this classifier reported every N-tail-needing shape as "supported"
    # regardless of gemm_n's value mod 4, silently over-counting coverage for any gemm_n
    # that isn't a multiple of 4 (not just the tiny gemm_n=1/3 cases above -- e.g. gemm_n=130
    # also needs N-relief and also fails this check, 130%4=2).
    n_tail_mod4_violation = ('N' in needs) and (gn % 4 != 0)

    is_1x1 = (d['y'] == 1 and d['x'] == 1)

    if d['direction'] == 'fwd':
        if n_tail_mod4_violation:
            return 'gap_n_mod4_fwd', (
                f"gm={gm},gn={gn},gk={gk}: N needs tail relief but gn={gn} is not a "
                f"multiple of 4 -- fwd's non-atomic vectorized-4-wide-store epilogue "
                f"requires this (confirmed unbuildable on real hardware), no config covers "
                f"it today ({prec})")
        if 'K' in needs:
            if is_1x1:
                # Phase 37: TDM's own descriptor covers M/N tail natively (tensor_dim1
                # rebuilt relative to the block offset, same mechanism Phase 31 already used
                # for K) -- any subset of M/N/K needed together is covered by ONE config
                # (`_tdm_mntail`, wmma_m_tail=wmma_n_tail=1 unconditionally -- a no-op mask
                # on shapes that don't actually need it, confirmed by hardware sanity
                # checks), no gm/gn%128==0 requirement left at all.
                if prec == 'int8':
                    return 'gap_config_int8_fwd_tdm', "mechanism (TDM K/M/N-tail) exists for fp16/bf16/fp32 only -- no int8 _tdm config"
                return 'supported_tail_fwd_tdm', f"needs {sorted(needs)} (K and/or M/N), 1x1 -- covered by _tdm_mntail ({prec})"
            # Phase 38: new non-TDM GEMM_K tail for multi-tap convs (TDM was never extended
            # past 1x1). fwd's A and B are BOTH the "hard case" (fixed row per lane, K read
            # contiguously in one shot -- unlike bwd's B, fwd's B is natural/untransposed),
            # so this reuses bwd's Phase 36 fine-grained per-dword AND-mask primitive for
            # both operands. Composes with the existing wmma_m_tail/wmma_n_tail EXEC-mask
            # mechanisms (independent load-time vs. post-load masking, confirmed to compose
            # via hardware validation) -- any subset of M/N/K is covered by _ktail (K alone)
            # or _mnktail (all three).
            if prec == 'int8':
                return 'gap_config_int8_fwd_ktail', "mechanism (non-TDM K-tail) exists for fp16/bf16/fp32 only -- no int8 _ktail/_mnktail config"
            return 'supported_tail_fwd_ktail', f"needs {sorted(needs)} (K and/or M/N), multi-tap -- covered by _ktail/_mnktail ({prec})"
        else:
            if prec == 'int8':
                return 'gap_config_int8_fwd_tail', f"needs {sorted(needs)} -- wmma_m_tail/wmma_n_tail exist for fp16/bf16/fp32 only, no int8 config"
            return 'supported_tail_fwd_mn', f"needs {sorted(needs)} -- covered by _mtail/_ntail/_mntail ({prec})"

    if d['direction'] == 'bwd':
        # Phase 36: N-tail and K-tail now exist for bwd (in addition to M-tail, Phase 26a) --
        # B's transposed addressing meant N-tail/K-tail needed a genuinely new fine-grained
        # per-dword masking primitive (not a port of fwd's), but the mechanism itself is
        # precision-generic (data_byte-driven, like the rest of this kernel). bf16/fp32
        # _ntail/_ktail/_mnktail configs (written and hardware-validated right after Phase 36
        # landed, exercising bf16's identical-to-fp16 elem_per_dword=2 path and fp32's
        # distinct elem_per_dword=1 path) closed what was briefly a config-only gap.
        if n_tail_mod4_violation:
            return 'gap_n_mod4_bwd', (
                f"gm={gm},gn={gn},gk={gk}: N needs tail relief but gn={gn} is not a "
                f"multiple of 4 -- bwd shares fwd's non-atomic vectorized-4-wide-store "
                f"epilogue (coalescing_store_wmma.py), same hardware constraint, no config "
                f"covers it today ({prec})")
        if prec == 'int8':
            return 'gap_config_int8_bwd_tail', f"needs {sorted(needs)} -- no int8 config exists for ANY bwd tail mechanism (not even pre-existing M-tail), and int8's elem_per_dword=4 masking case is wholly untested"
        return 'supported_tail_bwd_mnk', f"needs {sorted(needs)} -- covered by _mtail/_ntail/_ktail/_mnktail ({prec})"

    # wrw (post Phase 35: M/N/K-tail all exist in code; fp16/bf16/fp32 all have committed
    # tail-relief configs now, gsplit exists fp16/bf16/fp32 but not int8)
    if prec == 'int8':
        return 'gap_config_int8_wrw_tail', f"needs {sorted(needs)} -- wrw has no tail-relief config for int8 at all (and no int8 gsplit either)"
    if needs == {'M', 'N', 'K'}:
        note = f"needs all three (M+N+K) -- covered by _gsplit_mnktail, 64x64-tile-only (128x128 overflows 256 VGPRs by 1)"
    else:
        note = f"needs {sorted(needs)} -- covered by _mntail/_ktail/_gsplit_mntail/_gsplit_ktail/_gsplit_mnktail ({prec})"
    return 'supported_tail_wrw', note


CATEGORY_LABELS = {
    'supported_exact_fit': 'Supported today (exact tile fit, no relief needed)',
    'supported_tail_fwd_mn': 'Supported today (fwd M/N-tail)',
    'supported_tail_fwd_tdm': 'Supported today (fwd TDM K/M/N-tail, Phase 31/37)',
    'supported_tail_fwd_ktail': 'Supported today (fwd non-TDM K-tail, multi-tap, Phase 38)',
    'supported_tail_bwd_mnk': 'Supported today (bwd M/N/K-tail, Phase 36)',
    'supported_tail_wrw': 'Supported today (wrw M/N/K-tail, Phase 35)',
    'gap_layout_not_nhwc': 'GAP: non-NHWC layout',
    'gap_precision_unsupported': 'GAP: unsupported precision (int4 etc.)',
    'gap_depthwise': 'GAP: depthwise (architecturally out of scope)',
    'gap_invalid_group': 'GAP: malformed group count',
    'degenerate_zero_output': 'DEGENERATE: zero valid output positions (not a real conv)',
    'gap_config_int8_fwd_tdm': 'GAP: fwd int8 TDM config missing',
    'gap_config_int8_fwd_ktail': 'GAP: fwd int8 non-TDM K-tail config missing',
    'gap_config_int8_fwd_tail': 'GAP: fwd int8 M/N-tail config missing',
    'gap_config_int8_bwd_tail': 'GAP: bwd int8 tail config missing (any mechanism)',
    'gap_config_int8_wrw_tail': 'GAP: wrw int8 tail/gsplit config missing',
    'gap_n_mod4_fwd': 'GAP: fwd gemm_n needs tail but not a multiple of 4 (vectorized-store epilogue limit)',
    'gap_n_mod4_bwd': 'GAP: bwd gemm_n needs tail but not a multiple of 4 (vectorized-store epilogue limit)',
}

# Rough, qualitative effort tiers for the write-up.
EFFORT_TIER = {
    'gap_layout_not_nhwc': 'C (driver integration: wire existing transpose kernels into the WMMA dispatch path)',
    'gap_precision_unsupported': 'D (new kernel work: no gfx1250 WMMA int4 path exists)',
    'gap_depthwise': 'D (architectural: would need a fundamentally different codegen strategy)',
    'gap_invalid_group': 'n/a (malformed input, not a real gap)',
    'degenerate_zero_output': 'n/a (not a real conv, low priority even if closable)',
    'gap_config_int8_fwd_tdm': 'B (config file + validation only)',
    'gap_config_int8_fwd_ktail': 'B (config file + validation only)',
    'gap_config_int8_fwd_tail': 'B (config file + validation only)',
    'gap_config_int8_bwd_tail': 'C (config file + first-ever validation of the elem_per_dword=4 masking case for bwd)',
    'gap_config_int8_wrw_tail': 'D for gsplit (new mechanism, int8 gsplit doesn\'t exist) + B for tail (config only)',
    'gap_n_mod4_fwd': 'C (new epilogue masking granularity -- shared coalescing_store_wmma.py mechanism change, affects fwd+bwd)',
    'gap_n_mod4_bwd': 'C (new epilogue masking granularity -- shared coalescing_store_wmma.py mechanism change, affects fwd+bwd)',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fwd_file')
    ap.add_argument('bwd_file')
    ap.add_argument('wrw_file')
    ap.add_argument('--csv-out', default=None)
    ap.add_argument('--md-out', default=None)
    ap.add_argument('--examples-per-category', type=int, default=4)
    ap.add_argument('--assume-nhwc', action='store_true',
                     help="skip the layout check -- models 'every shape gets transposed to "
                          "NHWC by the caller' and reports pure GEMM-shape/mechanism coverage")
    args = ap.parse_args()

    all_shapes = []
    for path, direction in [(args.fwd_file, 'fwd'), (args.bwd_file, 'bwd'), (args.wrw_file, 'wrw')]:
        for shape in parse_file(path, direction):
            cat, note = classify(shape, assume_nhwc=args.assume_nhwc)
            shape['category'] = cat
            shape['note'] = note
            all_shapes.append(shape)

    print(f"Total shape entries parsed: {len(all_shapes)}", file=sys.stderr)

    # Dedup for reporting purposes (exact same shape, e.g. repeated across many N-sweeps
    # for the SAME classification-relevant fields, but N does affect gm/gk so keep n).
    dedup_key_fields = ['direction', 'precision', 'n', 'c', 'k', 'H', 'W', 'y', 'x', 'p', 'q',
                         'u', 'v', 'l', 'j', 'g', 'in_layout', 'fil_layout', 'out_layout']
    seen = set()
    distinct_shapes = []
    for s in all_shapes:
        key = tuple(s[f] for f in dedup_key_fields)
        if key not in seen:
            seen.add(key)
            distinct_shapes.append(s)
    print(f"Distinct shapes (dedup by full param tuple): {len(distinct_shapes)}", file=sys.stderr)

    if args.csv_out:
        import csv
        with open(args.csv_out, 'w', newline='') as f:
            fieldnames = dedup_key_fields + ['solver', 'ref_time_ms', 'category', 'note']
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for s in distinct_shapes:
                w.writerow(s)
        print(f"wrote {args.csv_out} ({len(distinct_shapes)} rows)", file=sys.stderr)

    # Aggregate: counts by direction x category, on ALL entries (reflects real corpus
    # weight/frequency) and on DISTINCT shapes (reflects shape-space diversity).
    counts_all = Counter((s['direction'], s['category']) for s in all_shapes)
    counts_distinct = Counter((s['direction'], s['category']) for s in distinct_shapes)
    precision_counts = Counter((s['direction'], s['category'], s['precision']) for s in distinct_shapes)

    examples = defaultdict(list)
    for s in distinct_shapes:
        key = (s['direction'], s['category'])
        if len(examples[key]) < args.examples_per_category:
            examples[key].append(s)

    lines = []
    lines.append("# gfx1250 WMMA coverage vs. a real MISA-solver-won shape corpus\n")
    lines.append(f"Parsed {len(all_shapes)} total entries ({len(distinct_shapes)} distinct shapes) "
                 f"from three real MIOpen solver-search traces, all shapes where a MISA-authored "
                 f"solver (`ConvAsmImplicitGemmGTCDynamic{{Fwd,Bwd,Wrw}}XdlopsNHWC`) won MIOpen's own "
                 f"solver search on the traced architecture.\n")

    for direction in ['fwd', 'bwd', 'wrw']:
        total_all = sum(v for (d, c), v in counts_all.items() if d == direction)
        total_distinct = sum(v for (d, c), v in counts_distinct.items() if d == direction)
        lines.append(f"\n## {direction} -- {total_all} entries, {total_distinct} distinct shapes\n")
        lines.append("| Category | Distinct shapes | All entries (corpus weight) | Effort to close |")
        lines.append("|---|---|---|---|")
        cats = sorted([c for (d, c) in counts_distinct if d == direction],
                      key=lambda c: -counts_distinct[(direction, c)])
        for cat in cats:
            label = CATEGORY_LABELS.get(cat, cat)
            n_distinct = counts_distinct[(direction, cat)]
            n_all = counts_all[(direction, cat)]
            effort = EFFORT_TIER.get(cat, '-' if cat.startswith('supported') else '?')
            lines.append(f"| {label} | {n_distinct} | {n_all} | {effort} |")

        lines.append(f"\n### Examples ({direction})\n")
        for cat in cats:
            if cat.startswith('supported'):
                continue
            exs = examples[(direction, cat)]
            if not exs:
                continue
            lines.append(f"\n**{CATEGORY_LABELS.get(cat, cat)}**:")
            for s in exs:
                lines.append(f"- `{s['precision']} n={s['n']} c={s['c']} k={s['k']} H={s['H']} W={s['W']} "
                             f"y={s['y']} x={s['x']} p={s['p']} q={s['q']} u={s['u']} v={s['v']} "
                             f"l={s['l']} j={s['j']} g={s['g']}` -- {s['note']} "
                             f"(ref: {s['ref_time_ms']}ms, {s['solver']})")

    output = "\n".join(lines)
    if args.md_out:
        with open(args.md_out, 'w') as f:
            f.write(output + "\n")
        print(f"wrote {args.md_out}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
