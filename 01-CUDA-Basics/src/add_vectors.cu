#include <iostream>

__global__ void add(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    int N = 100;
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    // Allocate memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy the data from host to device
    cudaMemcpy(d_a, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_B, size, cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and N threads
    add<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back from device to host
    cudaMemcpy(h_C, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < 10; i++)
    {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
