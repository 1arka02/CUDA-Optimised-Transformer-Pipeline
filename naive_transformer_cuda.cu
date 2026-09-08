
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<cuda_runtime.h>


__global__ void transposeKernel(const float* In, float* Out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        Out[c * rows + r] = In[r * cols + c];
    }
}


__global__ void matMulKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void softmaxKernel(const float* S, float* Softmax, int N) {
    int row = blockIdx.x; 

    if (row >= N) return;

    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    float thread_max = -1e30f; 
    for (int col = tid; col < N; col += blockDim.x) {
        thread_max = fmaxf(thread_max, S[row * N + col]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        thread_sum += expf(S[row * N + col] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    for (int col = tid; col < N; col += blockDim.x) {
        Softmax[row * N + col] = expf(S[row * N + col] - row_max) / row_sum;
    }
}

//Print Matrix
void printMatrix(float *mat,int row,int column){
    for(int i= 0;i<row;i++){
        for(int j = 0;j<column;j++){
            printf("%lf ",mat[i*column + j]);
        }
         printf("\n");
    }
}


int main() {
    int N = 4096; // Matrix dimension (NxN)
    size_t bytes = N * N * sizeof(float);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_KT = (float *)malloc(bytes);
    float *h_QKT = (float *)malloc(bytes);
    float *h_Softmax = (float *)malloc(bytes);
    float *h_Output = (float *)malloc(bytes);

    for(int i = 0; i< N*N; i++){
        h_Q[i] = 1;
        h_K[i] = 1;
        h_V[i] = 1;
    }

    // Device pointers
    float *d_Q, *d_K, *d_V, *d_KT, *d_QKT, *d_Softmax, *d_Output;

    // Allocate Device memory
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_KT, bytes);
    cudaMalloc(&d_QKT, bytes);
    cudaMalloc(&d_Softmax, bytes);
    cudaMalloc(&d_Output, bytes);

    // Copy input data from Host to Device
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    // Define 2D block and grid configurations for matrix ops
    dim3 threadsPerBlock2D(16, 16);
    dim3 blocksPerGrid2D((N + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                        (N + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);

    // 1. Execute Kernel: K^T Transpose
    transposeKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_K, d_KT, N, N);


    // 2. Execute Kernel: Q * K^T
    matMulKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_Q, d_KT, d_QKT, N, N, N);


    // 3. Execute Kernel: Row-wise Softmax using Reduction
    // Launch 1 block per row, threads per block = 32 (or nearest power of 2 >= N)
    int threadsPerBlockSoftmax = 32;
    size_t sharedMemBytes = threadsPerBlockSoftmax * sizeof(float);
    softmaxKernel<<<N, threadsPerBlockSoftmax, sharedMemBytes>>>(d_QKT, d_Softmax, N);


    // 4. Execute Kernel: Softmax * V
    matMulKernel<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_Softmax, d_V, d_Output, N, N, N);


    // Copy intermediate and final results back to Host
    cudaMemcpy(h_KT, d_KT, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_QKT, d_QKT, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Softmax, d_Softmax, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Output, d_Output, bytes, cudaMemcpyDeviceToHost);

    // Print Results
    printf("H_kT\n");
    printMatrix( h_KT, N, N);
    printf("H_QKT\n");
    printMatrix( h_QKT, N, N);
    printf("H_softmax\n");
    printMatrix( h_Softmax, N, N); 
    printf("Output\n");
    printMatrix( h_Output, N, N);

    // Free Memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_KT);
    cudaFree(d_QKT);
    cudaFree(d_Softmax);
    cudaFree(d_Output);


    return 0;
}
