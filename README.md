# CUDA-Optimized Transformer Self-Attention Pipeline

A CUDA implementation of a simplified Transformer self-attention mechanism, with a shared-memory tiled transpose kernel that replaces the naive baseline to improve end-to-end pipeline performance.

## Overview

Given input matrices **Q**, **K**, and **V**, the pipeline computes scaled dot-product self-attention:

1. **Transpose** — compute Kᵀ
2. **Matrix multiply** — compute S = Q × Kᵀ
3. **Row-wise softmax** — normalize each row of S using a numerically stable reduction:

   ```
   P_ij = exp(S_ij − M_i) / Σ_j exp(S_ij − M_i)
   ```

   where `M_i` is the maximum value in row *i*.
4. **Matrix multiply** — compute the final output as Softmax(S) × V

Each step is implemented as its own CUDA kernel: a transpose kernel, a matrix-multiplication kernel (used twice, for QKᵀ and Softmax·V), and a softmax kernel using a reduction for the row-wise max and sum.

### Example

```
Q = [[1, 2],       K = [[5, 6],       V = [[1, 2],
     [3, 4]]            [7, 8]]            [3, 4]]

Kᵀ =            QKᵀ =              Softmax(QKᵀ) =              Output =
5 7             17 23              0.00247     0.99753          2.995  3.995
6 8             39 53              0.00000083  0.99999917       3.000  4.000
```

Matrices are generated with sequential values for easy correctness verification against a known-good reference.

## Optimization: Shared-Memory Tiled Transpose

The baseline transpose kernel writes to global memory in an uncoalesced pattern, which becomes a bottleneck as matrix size grows. This project replaces it with a **shared-memory tiled transpose kernel**: each thread block loads a tile into on-chip shared memory (with padding to avoid bank conflicts), then writes it back out in a fully coalesced pattern — converting scattered global-memory writes into contiguous ones.

## Results (N = 4096, benchmarked on Google Colab GPUs)

### End-to-End Pipeline

| Statistic                        | Original  | Optimized |
|-----------------------------------|-----------|-----------|
| Transpose execution time (ms)     | 1.0000    | 0.6858    |
| End-to-end execution time (ms)    | 534       | 362       |
| HtoD transfer time (ms)           | 51.1199   | 43.5168   |
| HtoD data transferred (MB)        | 192.0000  | 192.0000  |
| **End-to-end speedup**            | 1.00×     | **1.47×** |

### Kernel-Level Breakdown (via `nsys`)

**Original pipeline**

| Kernel           | Time (%) | Total Time (ns)  | Instances |
|-------------------|----------|-------------------|-----------|
| matMulKernel       | 99.8     | 1,000,087,846     | 2         |
| transposeKernel    | 0.1      | 1,079,770         | 1         |
| softmaxKernel      | 0.1      | 753,691           | 1         |

**Optimized pipeline**

| Kernel                     | Time (%) | Total Time (ns) | Instances |
|-----------------------------|----------|------------------|-----------|
| matMulKernel                 | 99.2     | 126,351,504      | 2         |
| softmaxKernel                | 0.6      | 752,124          | 1         |
| transposeKernel (Optimized)  | 0.2      | 251,422          | 1         |

### Memory Transfer

| Version   | Operation     | Time (%) | Total Time (ns) | Count |
|-----------|----------------|----------|------------------|-------|
| Original  | memcpy DtoH    | 81.7     | 244,370,676      | 4     |
| Original  | memcpy HtoD    | 18.3     | 54,917,593       | 3     |
| Optimized | memcpy DtoH    | 83.0     | 250,515,438      | 4     |
| Optimized | memcpy HtoD    | 17.0     | 51,433,486       | 3     |

Total data transferred is identical across both versions (268.4 MB DtoH, 201.3 MB HtoD) since transpose optimization affects compute pattern, not data volume.

## Key Takeaway

The tiled transpose kernel delivers a **1.4–4.3× improvement** over the naive baseline at the kernel level (depending on host-side vs. device-side `nsys` timing), by converting an uncoalesced global-memory write pattern into fully coalesced reads and writes via a padded on-chip tile.

At the pipeline level, however, the transpose is a small fraction of total runtime — matrix multiplication dominates (>99% of kernel time in both versions). This means transpose optimization matters most when:
- the transpose is repeated frequently,
- matrices are very large, or
- the matrix multiplication itself is already highly optimized (e.g., via tensor cores or cuBLAS) — at which point transpose's relative share of total time grows, and optimizing it has more impact.

## Repository Structure

```
.
├── src/                  # CUDA source files (kernels + host driver)
├── report/               # Performance analysis report (PDF)
└── README.md
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (`nvcc`)
- NVIDIA Nsight Systems (`nsys`) for profiling (optional)

## Building & Running

```bash
nvcc -O3 -o self_attention src/main.cu
./self_attention
```

## Profiling

```bash
nsys profile --stats=true ./self_attention
```
