# Simple perf testing for SIMD delayed operations

We do a little test where by we run through two vectors and add their product and quotient.
So, basically three operations per pair of vector elements, which should be interesting enough for the test to not be completely memory-bound.
We disable GCC's autovectorization code to see the effect of the SIMD intrinsics.
Currently, the tests below are run on Intel i7 with GCC 11.4.

## `double` results

Small improvement with a single thread:

```sh
$ ./build/testing -t 1       # scalar
Access time: 405, yielding 7.27009e+06

$ ./build/testing -t 1 -s    # AVX2
Access time: 357, yielding 7.27009e+06
```

Largely lost with parallelization:

```sh
$ ./build/testing -t 4       # scalar
Access time: 183, yielding 7.27009e+06

$ ./build/testing -t 4 -s    # AVX2
Access time: 186, yielding 7.27009e+06
```

## `float` results

Might as well test `float`s as well, not that I use them much.
Again, a modest improvement with a single thread:

```sh
$ ./build/testing -t 1 -f    # scalar
Access time: 310, yielding -563723

$ ./build/testing -t 1 -s -f # AVX2
Access time: 235, yielding -563723
```

More or less the same with parallelization:

```sh
$ ./build/testing -t 4 -f    # scalar
Access time: 110, yielding -563728

$ ./build/testing -t 4 -s -f # AVX2
Access time: 87, yielding -563728
```

## Comments

We see modest benefits for typical **tatami** applications where there isn't a lot of heavy compute.
It deteriorates with more high-level parallelization where (presumably) memory bottlenecks dominate.

Mind you, this comparison involves disabling autovectorization explicitly. 
If we just use the default flags, GCC is able vectorize this simple routine, at which point the differences disappear.

In practice, it seems that most of the relevant code in **tatami** (and elsewhere, e.g., **libscran**) is trivially vectorizable,
This is based on testing of modern compilers with a relevant `-march` setting on godbolt.org.
The same testing also revealed few missed vectorization opportunities:

- Gather and scatter commands when retrieving and setting sparse indices.
  This cannot be auto-vectorized due to the theoretical possibility of repeated indices.
- Certain `<cmath>` operations cannot be auto-vectorized due to errno-related side-effects.
  This requires some additional flags like `-fno-math-errno` for `sqrt()` (which is acceptable)
  or `-ffast-math` for `log()` (which is not) to convince the compiler to auto-vectorize.
- Reduction operations on floats don't get autovectorized without `-ffast-math`, obviously.
  Interestingly, integer reductions are auto-vectorized, though these are pretty rare in my code.

I spent some time testing `#pragma omp simd`, hoping that it could instruct the compiler to ignore vector dependencies without having to write vector instrinsics.
This manages to cajole GCC into using gather/scatter commands for sparse indices but doesn't help with the other missed opportunities.
For general use, it is at best unnecessary as autovectorization occurs in most cases anyway;
and at worst, it might be a pessimisation if it [overrides the compiler's cost model](https://developers.redhat.com/articles/2023/12/08/vectorization-optimization-gcc)
and forces the use of, e.g., slow instructions, reduced clock speeds.
I also found out that MSVC enables [fast floating-math with this pragma](https://devblogs.microsoft.com/cppblog/simd-extension-to-c-openmp-in-visual-studio/),
which is not something I want in general.
All in all, it didn't seem worth the trouble.
Instead, I ended up writing a cross-compiler macro in [`SUBPAR_VECTORIZABLE`](https://github.com/LTLA/subpar) to indicate that loop iterations are independent.
This encourages auto-vectorization while still respecting the compiler's cost calculations.

The alternative to OpenMP SIMD would be to use a third-party library. 
For example, a library like Eigen implements its own vectorized math functions that do not relyi on dangerous compiler flags.
This is a no-go for the core **tatami** packages as it violates my standing rule against external dependencies,
but you could imagine implementing vectorized versions of delayed operations in a separate library that accepts the dependency burden.
(In the most common case of a log-transformed normalized matrix, a separate helper operation would be desirable anyway,
as this would allow calculation of the `log1p(x / sf) / log(2)` operation in one pass instead of 3.)

Another practical consideration is that applications relying on vector intrinsics are difficult to deploy.
I've run into problems with `-march=native` on heterogeneous clusters before,
and I would guess that this is exacerbated once the code relies on more modern intrinsics (e.g., AVX512).
This is probably why [Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Writing-portable-packages) explicitly recommends against setting `-march`.
I suppose I could just compile with something like GCC's function multi-versioning but that's such a chore to specify all of the instruction sets.

## Build instructions

Just use the usual CMake process on a decently modern Intel CPU:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
