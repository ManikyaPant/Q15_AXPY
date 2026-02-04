#!/bin/bash

clang --target=riscv64-linux-gnu \
    -march=rv64gcv \
    --sysroot=/opt/riscv/sysroot \
    --gcc-toolchain=/opt/riscv \
    -O3 \
    -static \
    solution.c

