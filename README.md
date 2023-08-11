# Simple perf testing for SIMD delayed operations

## Initial results

```sh
$ ./build/testing -t 1       # scalar
Access time: 242, yielding 7.27553e+06

$ ./build/testing -t 1 -s    # AVX2
Access time: 192, yielding 7.27553e+06
```

```sh
$ ./build/testing -t 4       # scalar
Access time: 74, yielding 7.27553e+06

$ ./build/testing -t 4 -s    # AVX2
Access time: 62, yielding 7.27553e+06
```

## What about floats?

```sh
$ ./build/testing -t 1 -f    # scalar
Access time: 203, yielding -556276

$ ./build/testing -t 1 -s -f # AVX2
Access time: 137, yielding -556276
```

```sh
$ ./build/testing -t 4 -f    # scalar
Access time: 55, yielding -556271

$ ./build/testing -t 4 -s -f # AVX2
Access time: 40, yielding -556271
```

## Build instructions

Just use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
