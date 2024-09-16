# CUDA Chapter 3: Multidimensional Grids and Data

## Key Concepts

### 1. Multidimensional Grid Organization

- **CUDA grids and blocks** can be up to 3-dimensional, providing flexibility to map threads to multidimensional data like matrices or images.
- **Thread identification** is done using built-in variables like `blockIdx` and `threadIdx` to allow each thread to identify itself.
- **Grid and block dimensions** are specified in the kernel execution configuration using the `dim3` type.
- Dimensions can be accessed via the `gridDim` and `blockDim` variables.

### 2. Thread Hierarchy

- A **grid** consists of multiple blocks, and each **block** contains multiple threads.
- All threads in the grid execute the same kernel function, with each thread using its unique coordinates (`blockIdx`, `threadIdx`) to process data.

### 3. Execution Configuration

- Execution configuration is set using the `<<<gridDim, blockDim>>>` syntax, where `gridDim` defines the number of blocks and `blockDim` defines the number of threads per block.
- CUDA supports up to 3D structures using the `dim3` type.
- For simple 1D cases, arithmetic expressions can be used to set the dimensions.

### 4. Dimension Limits

- **Grid dimension limits**:
  - `gridDim.x`: 1 to \(2^{31} - 1\) (older devices: 1 to \(2^{16} - 1\))
  - `gridDim.y` and `gridDim.z`: 1 to \(2^{16} - 1\)
- **Block limits**: The total number of threads per block is limited to 1024 threads.

---

## Mapping Threads to Multidimensional Data

### 1. Linearization of Multidimensional Arrays

- CUDA uses **row-major order** for multidimensional arrays, similar to C.
- For efficient processing, multidimensional arrays need to be **linearized**:
  - For 2D arrays: `row * width + col`
  - For 3D arrays: `plane * m * n + row * m + col`

### 2. Thread-to-Data Mapping

- A common practice is to have a **one-to-one mapping** between threads and data elements.
- This is achieved by calculating data indices using the `blockIdx`, `blockDim`, and `threadIdx` built-in variables, which represent the thread's position in the grid and block.

---

## Examples

### 1. Color to Grayscale Conversion

- A **2D thread grid** can be used to represent a 2D image, where each thread is responsible for processing a single pixel.
- **Linearized index** for each pixel: `grayOffset = row * width + col`
- The RGB to grayscale conversion formula used is:  
  `L = 0.21R + 0.72G + 0.07B`

### 2. Image Blur

- This example involves **neighborhood operations**, where each thread calculates the average color of the surrounding pixels to blur the image.
- The kernel uses **nested loops** to process the patch of neighboring pixels.
- Special care is taken to handle **boundary conditions** for edge pixels.

### 3. Matrix Multiplication

- Matrix multiplication is a fundamental operation in **Basic Linear Algebra Subprograms (BLAS)**.
- Each thread is responsible for calculating a single output element, which is the **dot product** of a row from matrix M and a column from matrix N.
- Linearized index for matrices:
  - For matrix M: `row * Width + k`
  - For matrix N: `k * Width + col`

---

## Key Programming Techniques

1. **Index checking**: Use `if` statements to ensure thread indices are within valid ranges.
2. **Array linearization**: Efficiently access elements in multidimensional arrays by calculating linear indices.
3. **Boundary condition handling**: Take care to handle edge cases, especially in neighborhood operations.
4. **Reduction operations**: Use local variables within each thread to accumulate values during reduction operations (e.g., matrix multiplication).

---

## Limitations and Considerations

- **Grid size limitations**: The size of the grid is limited by the number of available blocks and threads per block.
- For **large-scale problems**, one approach is to divide the data into smaller submatrices or have threads process multiple data elements to fit within hardware limits.

---
