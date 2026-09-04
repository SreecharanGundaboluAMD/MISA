{
  "task": "W-5: Un-gate ds_load_tr_b for wrw_streamk",
  "status": "complete",
  "validation": {
    "ds_load_tr_b=1 + wrw_streamk=1": "all 6 runs valid:y (3 shapes × 2 independent process launches each)",
    "shape_1": {
      "description": "128x1024x17x17x1024 (1x1, grid=68)",
      "before_dstrb0": {
        "run1": {
          "wrw0": 295.176,
          "wrw1": 269.514,
          "total": 564.69
        },
        "run2": {
          "wrw0": 293.899,
          "wrw1": 273.527,
          "total": 567.426
        },
        "run3": {
          "wrw0": 294.447,
          "wrw1": 267.948,
          "total": 562.395
        },
        "avg_total": 564.837
      },
      "after_dstrb1": {
        "run1": {
          "wrw0": 268.88,
          "wrw1": 292.461,
          "total": 561.341
        },
        "run2": {
          "wrw0": 268.07,
          "wrw1": 293.466,
          "total": 561.536
        },
        "run3": {
          "wrw0": 268.512,
          "wrw1": 294.217,
          "total": 562.729
        },
        "avg_total": 561.869
      },
      "impact": "-2.968 TFLOP/s (-0.53%) — noise"
    },
    "shape_2": {
      "description": "256x2048x14x14x2048 (1x1, grid=224)",
      "before_dstrb0": {
        "run1": {
          "wrw0": 228.135,
          "wrw1": 213.084,
          "total": 441.219
        },
        "run2": {
          "wrw0": 228.87,
          "wrw1": 213.175,
          "total": 442.045
        },
        "run3": {
          "wrw0": 228.463,
          "wrw1": 212.41,
          "total": 440.873
        },
        "avg_total": 441.379
      },
      "after_dstrb1": {
        "run1": {
          "wrw0": 228.542,
          "wrw1": 224.515,
          "total": 453.057
        },
        "run2": {
          "wrw0": 228.135,
          "wrw1": 223.787,
          "total": 451.922
        },
        "run3": {
          "wrw0": 229.044,
          "wrw1": 223.88,
          "total": 452.924
        },
        "avg_total": 452.634
      },
      "impact": "+11.255 TFLOP/s (+2.55%)"
    }
  },
  "changes": [
    {
      "file": "python/igemm/igemm_base.py",
      "lines": "904-905",
      "change": "Removed `and not tunable_dict.get('wrw_streamk', 0)` from `_dstrb_default`; updated comment block"
    },
    {
      "file": "driver/igemm_gtc_base.h",
      "lines": "399-401",
      "change": "Removed `!tunable.wrw_streamk` from C++ default; synced comment"
    }
  ],
  "commit": "1b95803 [WMMA][gfx1250] W-5: Un-gate ds_load_tr_b for wrw_streamk",
  "findings_doc": "docs/gfx1250_w5_streamk_dstrb_ungate.md",
  "performance_summary": "Shape 1: -0.98% (noise); Shape 2: +2.69% (measurable, driven by wrw1 64x64 tile +5-8% benefit from hardware transpose-load removal of 128-packing VALU + 8 waits/iteration)"
}