# CuTe Ampere DGEMM

The exact operation implemented for now is NT DGEMM: `(M,K):(K,1) @ (N,K):(K,1) = (M,N):(1, M)`, B is transposed for simplicity
This repo doesn't work on RTX Ampere: "Note that GA10x GPUs do not include Tensor Core acceleration for double-precision (FP64) operations, as provided in A100." -- NVIDIA Ampere RTX whitepaper


<p align="center">
  <img src="./bench.png" />
</p>

TESTED ON "A100 DRIVE":
- 1140 Mhz (`nvidia-smi -q -d CLOCK` Clocks > SM, mine is downclocked don't worry about it)
- 96 SMs
- 384 Tensor cores (4 TC/SM), ampere (DC /!\) TC can do 16FMA/cycle

Theoretical flops: `1140*1e6 * 384 * 16*2 / 1e12 = 14 TFlops` (double)

## Requirements
- CUDA TOOLKIT >= 12.9 (for nvcc and cublas)
- cutlass `git clone https://github.com/NVIDIA/cutlass.git`

## Usage
```shell
make baseline
./baseline > baseline.csv
./plot
```
