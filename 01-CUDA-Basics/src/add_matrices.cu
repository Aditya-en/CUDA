#include <iostream>
#include <cuda_runtime.h>

#define N 100

// Helper function for checking CUDA errors
#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

__global__ void Add_matrices(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        int idx = i * n + j;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    float H_A[N][N], H_B[N][N], H_C[N][N];

    // Initialize the host matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            H_A[i][j] = 1;
            H_B[i][j] = 2;
        }
    }

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory in device
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    cudaCheckError();

    // Move the matrices to device
    cudaMemcpy(d_A, H_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_B, H_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    Add_matrices<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaCheckError();

    // Synchronize to check for any kernel launch errors
    cudaDeviceSynchronize();
    cudaCheckError();

    // Copy the result back to the host
    cudaMemcpy(H_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the results
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << H_C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}