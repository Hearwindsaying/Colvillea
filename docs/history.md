# Background and History
This is a framework stemed/forked from my personal research renderer.
Initially, both of them would like to pursue high performance GPU rendering and support common lights, materials and BSDFs models.
But they could diverge greatly afterwards: my research renderer is mainly for research/experimention purposes rather than production rendering. So production renderer features would not be considered in high priority (compared to state-of-the-art rendering algorithms). It also does not care about USD/Hydra delegate that much but you certainly want this.

Credits of rendering algorithms go directly to Mitsuba, PBRT and LuxCoreRender.
Some ideas are also inspired by my prior working experience (e.g. Shading normal auto adapation and MIS-correct invisible area lights).

Many thanks for LightHouse renderer!

# FAQs
## Why starting from scratch instead of forking one from cycles/luxcore?
Forking a fully-featured production renderer gives you benefits that lots of features are already implemented and you do not need to worry about too much...
However, a fully-featured production renderer has a rather complex codebase and features which might not be suitable for your specific use case and they could suffer from lots of historical issues, which you certainly do not want to receive and fix.

For example, LuxCoreRender is great for physically correctness but it is a bit slow compared to Cycles. It also suffers from duplicate kernel and host side implementation issue (for the same feature, you have to write code separately for CPU and GPU and implement twice).

As for Cycles, it is great for animation/film but is less accurate for production rendering.

So our idea is to start from scratch, and go modern directly (no antique CMake, C++ etc.).
When you have your own code written by yourself, you will know what is going on and every hidden aspects of your renderer.

## Third party libraries and OWL?
Many renderer comes with their own low level infrastructure and base classes.
This is great for consistency but developing these low-level libraries is already complicated by itself.
You also need to put lots of efforts to ensure its usability and correctness.
Our goal is to use existing (easy-to-use) tools to build a renderer quickly.

So use some cool libraries save us time from engineering, and it also keeps your code neat and concise (think about how many lines of code we have written and OptiX kernel is really a small part of that, thanks to Ingo Wald's OWL).

## Performance?
Current implementation is quite efficient and it is 2x-10x faster than my old renderer framework.
I have not yet run profiler so it could be further optimized!