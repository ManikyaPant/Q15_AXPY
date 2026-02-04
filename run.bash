#!/bin/bash
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./a.out $1