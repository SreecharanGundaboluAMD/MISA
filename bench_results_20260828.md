| Direction | Shape (c,H,W,k,y×x) | Config | MISA (ms) | MIOpen/gfx950 (ms) | MIOpen/gfx1250 (ms) | vs gfx950 | vs gfx1250 | MIOpen/gfx1250 solver |
|---|---|---|---|---|---|---|---|---|
| fwd | 256,1,1,16,1x1 | mntail | 0.02300 | 0.00621 | 0.00862 | 3.70x slower | 2.67x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 512,1,1,32,1x1 | mntail | 0.02900 | 0.00800 | 0.00971 | 3.63x slower | 2.99x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 32,1,1,512,1x1 | mtail | 0.01900 | 0.00629 | 0.00847 | 3.02x slower | 2.24x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 96,120,160,48,1x1 | ntail_direct | 0.05800 | 0.03997 | 0.05205 | 1.45x slower | 1.11x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,512,1x1 | mtail | 0.02800 | 0.02189 | 0.02047 | 1.28x slower | 1.37x slower | 220/ConvHipConv |
| fwd | 256,30,40,128,1x1 | mtail | 0.01700 | 0.01366 | 0.01492 | 1.24x slower | 1.14x slower | 220/ConvHipConv |
| fwd | 192,60,80,64,1x1 | base_direct | 0.02700 | 0.02282 | 0.03615 | 1.18x slower | 1.34x faster | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 64,60,80,128,1x1 | base | 0.02200 | 0.01824 | 0.02039 | 1.21x slower | 1.08x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 512,30,40,128,1x1 | mtail_direct | 0.02400 | 0.01953 | 0.01995 | 1.23x slower | 1.20x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 24,240,320,128,1x1 | tdm | 0.18100 | 0.17973 | 0.16886 | 1.01x slower | 1.07x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 256,60,80,64,1x1 | base_direct | 0.03100 | 0.02730 | 0.03960 | 1.14x slower | 1.28x faster | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,128,1x1 | mtail_direct | 0.01200 | 0.01112 | 0.01307 | 1.08x slower | 1.09x faster | 220/ConvHipConv |
| fwd | 64,60,80,256,1x1 | base | 0.04200 | 0.03168 | 0.03714 | 1.33x slower | 1.13x slower | 220/ConvHipConv |
| fwd | 192,120,160,48,1x1 | ntail_direct | 0.08800 | 0.08065 | 0.07747 | 1.09x slower | 1.14x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| fwd | 128,30,40,128,3x3 | mtail | 0.03900 | 0.03575 | 0.03641 | 1.09x slower | 1.07x slower | 220/ConvHipConv |
| fwd | 128,120,160,128,3x3 | base | 0.36600 | 0.36303 | 0.32842 | 1.01x slower | 1.11x slower | 220/ConvHipConv |
| fwd | 48,120,160,128,1x1 | tdm | 0.06300 | 0.06297 | 0.05992 | 1.00x slower | 1.05x slower | 137/ConvHipImplicitGemmGroupFwdXdlops |
| bwd | 512,1,1,32,1x1 | mtail_direct | 0.01800 | 0.00939 | 0.00945 | 1.92x slower | 1.90x slower | 220/ConvHipConv |
| bwd | 128,30,40,512,1x1 | mtail | 0.04300 | 0.02783 | 0.01857 | 1.54x slower | 2.32x slower | 220/ConvHipConv |
| bwd | 128,30,40,128,3x3 | mtail | 0.08500 | 0.05621 | 0.02388 | 1.51x slower | 3.56x slower | 220/ConvHipConv |
| bwd | 64,60,80,256,1x1 | base_direct | 0.04300 | 0.03461 | 0.04199 | 1.24x slower | 1.02x slower | 220/ConvHipConv |
| bwd | 512,30,40,128,1x1 | mtail | 0.04100 | 0.03528 | 0.02179 | 1.16x slower | 1.88x slower | 220/ConvHipConv |
| bwd | 128,120,160,128,3x3 | base | 0.59600 | 0.50721 | 0.31830 | 1.18x slower | 1.87x slower | 220/ConvHipConv |
| bwd | 256,30,40,128,1x1 | mtail_direct | 0.02700 | 0.02428 | 0.01395 | 1.11x slower | 1.93x slower | 220/ConvHipConv |
| bwd | 64,60,80,128,1x1 | base | 0.02700 | 0.02647 | 0.02664 | 1.02x slower | 1.01x slower | 155/ConvHipImplicitGemmGroupBwdXdlops |
| bwd | 192,60,80,64,1x1 | base | 0.03800 | 0.03852 | 0.03283 | 1.01x faster | 1.16x slower | 220/ConvHipConv |
| bwd | 128,30,40,128,1x1 | mtail | 0.01700 | 0.01779 | 0.01274 | 1.05x faster | 1.33x slower | 220/ConvHipConv |
| bwd | 256,60,80,64,1x1 | base | 0.04200 | 0.04617 | 0.03716 | 1.10x faster | 1.13x slower | 220/ConvHipConv |
| wrw | 128,30,40,128,3x3 | gsplit | 0.42100 | 0.08654 | 0.06740 | 4.86x slower | 6.25x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,120,160,128,3x3 | gsplit | 2.18800 | 0.66439 | 0.41365 | 3.29x slower | 5.29x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 192,60,80,64,1x1 | gsplit | 0.15100 | 0.05720 | 0.00561 | 2.64x slower | 26.93x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,30,40,128,1x1 | gsplit | 0.08600 | 0.03579 | 0.02838 | 2.40x slower | 3.03x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 512,30,40,128,1x1 | gsplit | 0.09800 | 0.04728 | 0.04997 | 2.07x slower | 1.96x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,512,1x1 | gsplit | 0.09800 | 0.05202 | 0.05295 | 1.88x slower | 1.85x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 128,30,40,128,1x1 | gsplit | 0.03500 | 0.02684 | 0.02198 | 1.30x slower | 1.59x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,128,1x1 | gsplit | 0.06300 | 0.05149 | 0.04026 | 1.22x slower | 1.56x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 64,60,80,256,1x1 | gsplit | 0.06700 | 0.05810 | 0.05214 | 1.15x slower | 1.29x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |
| wrw | 256,60,80,64,1x1 | gsplit | 0.06500 | 0.05890 | 0.05217 | 1.10x slower | 1.25x slower | 156/ConvHipImplicitGemmGroupWrwXdlops |

Summary (avg ratio, MISA/MIOpen -- >1 means MISA is slower):
- fwd: vs gfx950 avg 1.57x, vs gfx1250 avg 1.34x (17 shapes)
- bwd: vs gfx950 avg 1.23x, vs gfx1250 avg 1.74x (11 shapes)
- wrw: vs gfx950 avg 2.19x, vs gfx1250 avg 5.10x (10 shapes)
