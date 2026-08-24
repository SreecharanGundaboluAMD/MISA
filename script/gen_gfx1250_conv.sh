#!/bin/sh
# call this at top of generator
# correctness-first WMMA milestone: fwd/fp16/nhwc only (1x1, stride1, no-pad GEMM case)
OUT=out
KERNELS=igemm_gtc_wmma_nhwc_gfx1250

rm -rf $KERNELS ; mkdir $KERNELS
mkdir -p $KERNELS/fwd_fp16

python3 igemm_codegen.py -s config/igemm_fwd_gtc_gfx1250_nhwc_fp16.config ; cp $OUT/*.s $OUT/*.inc $KERNELS/fwd_fp16
