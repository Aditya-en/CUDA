#include <iostream>
#include <cuda_runtime.h>

#define N 100

__global__ void dot_product(float *a, float *b, float *c)
{
    int i = threadIdx.x;
    atomicAdd(c, a[i] * b[i]);
}

int main()
{
    // allocate and initialize the host arrays
    float *H_A, *H_B;
    float H_C = 0;
    H_A = (float *)malloc(N * sizeof(float));
    H_B = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        H_A[i] = 0.1f;
        H_B[i] = 0.2f;
    }

    // device pointers
    float *d_a, *d_b, *d_c;

    // allocate memory on device
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, sizeof(float));

    // move the data to device
    cudaMemcpy(d_a, H_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, H_B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &H_C, sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel
    dot_product<<<1, N>>>(d_a, d_b, d_c);

    // synchronize the device
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // move the result to host
    cudaMemcpy(&H_C, d_c, sizeof(float), cudaMemcpyDeviceToHost);

    // print the result
    std::cout << "Dot product result: " << H_C << std::endl;
    std::cout << "Vector Multiplication done successfully" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(H_A);
    free(H_B);

    return 0;
}