#include <iostream>
#include <cuda_runtime.h>
#define N 20

__global__ void Add_matrices(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < N && j < N)
    {
        C[i][j] = A[i][j] + B[i][j];
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
            H_A[i][j] = i + j;
            H_B[i][j] = i - j;
        }
    }

    // Device pointers
    float(*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Allocate memory in device
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Move the matrices to device
    cudaMemcpy(d_A, H_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, H_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 blockSize(N, N);
    Add_matrices<<<1, blockSize>>>(d_A, d_B, d_C);

    // Copy the result back to the host
    cudaMemcpy(H_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

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
