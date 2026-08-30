| Direction | Shape (c,H,W,k,y×x) | Config | MISA (ms) | MIOpen/gfx950 (ms) | MIOpen/gfx1250 (ms) | vs gfx950 | vs gfx1250 | MIOpen/gfx1250 solver |
|---|---|---|---|---|---|---|---|---|
| wrw | 128,30,40,128,3x3 | master | 0.03200 | 0.08654 | 0.06740 | 2.70x faster | 2.11x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,120,160,128,3x3 | master | 0.30500 | 0.66439 | 0.41365 | 2.18x faster | 1.36x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 192,60,80,64,1x1 | master | 0.03800 | 0.05720 | 0.00561 | 1.51x faster | 6.78x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,30,40,128,1x1 | master | 0.02700 | 0.03579 | 0.02838 | 1.33x faster | 1.05x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 512,30,40,128,1x1 | master | 0.03700 | 0.04728 | 0.04997 | 1.28x faster | 1.35x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,512,1x1 | master | 0.03800 | 0.05202 | 0.05295 | 1.37x faster | 1.39x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,128,1x1 | master | 0.01900 | 0.02684 | 0.02198 | 1.41x faster | 1.16x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,128,1x1 | master | 0.02700 | 0.05149 | 0.04026 | 1.91x faster | 1.49x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,256,1x1 | master | 0.04500 | 0.05810 | 0.05214 | 1.29x faster | 1.16x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,60,80,64,1x1 | master | 0.04500 | 0.05890 | 0.05217 | 1.31x faster | 1.16x faster | 156/ConvHipImplicitGemmGroupWrwXdlops |

Summary (avg ratio, MISA/MIOpen -- >1 means MISA is slower):
- wrw: vs gfx950 avg 0.65x, vs gfx1250 avg 1.37x (10 shapes)
