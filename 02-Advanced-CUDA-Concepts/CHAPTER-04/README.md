# CUDA Chapter 4: GPU Architecture and Execution Model

## 4.1 Architecture of a Modern GPU

- **GPU Design**: Optimized for parallel processing, handling thousands of threads simultaneously.
- **Streaming Multiprocessors (SMs)**: GPUs consist of multiple SMs, each executing groups of threads in parallel.
- **Memory Hierarchy**:
  - **Global Memory**: Accessible by all threads but slow compared to shared memory.
  - **Shared Memory**: On-chip memory, fast access, used for intra-block communication.
  - **Registers**: Private to each thread, used for fast computation.

## 4.2 Block Scheduling

- **Block Scheduling**: GPU schedules thread blocks to be executed on available SMs.
- **Dynamic Scheduling**: Blocks are assigned dynamically to SMs as soon as resources become available.
- **Concurrency**: Multiple thread blocks can execute concurrently on a single SM, depending on the available resources.

## 4.3 Synchronization and Transparent Scalability

- **Thread Synchronization**: Threads within the same block can synchronize using `__syncthreads()`, ensuring all threads reach a point before continuing.
- **Scalability**:
  - **Transparent Scalability**: CUDA’s ability to distribute workload across available SMs, independent of hardware specifics.
  - **Grid-Block Scalability**: Grid and block size can be tuned for different GPU architectures to scale applications.

## 4.4 Warps and SIMD Hardware

- **Warps**: Threads are grouped into warps (typically 32 threads) and executed in lockstep.
- **Single Instruction, Multiple Data (SIMD)**: GPUs use SIMD to execute the same instruction across all threads in a warp.
- **Warp Execution**: Each warp executes independently, though instructions within the warp are issued simultaneously.

### Warp Execution Mechanism:

- **Lockstep Execution**: All threads in a warp execute the same instruction, making execution highly efficient but prone to divergence if threads take different paths.

## 4.5 Control Divergence

- **Control Divergence**: Occurs when threads within a warp follow different execution paths (e.g., due to `if` statements).
- **Handling Divergence**:
  - GPU splits the warp into subsets of threads (warps) executing the different paths sequentially.
  - Performance Impact: Divergence reduces efficiency as not all threads execute in parallel.
- **Avoiding Divergence**: Minimize conditional branching to avoid splitting warps.

## 4.6 Warp Scheduling and Latency Tolerance

- **Warp Scheduling**: Warps are scheduled in a round-robin fashion, with inactive warps waiting for resources.
- **Latency Tolerance**: GPUs hide memory access latency by switching to another warp while waiting for memory operations to complete.
  - **Occupancy**: High warp occupancy (many active warps) improves latency hiding and maximizes parallelism.

## 4.7 Resource Partitioning and Occupancy

- **Resource Partitioning**: Shared memory, registers, and threads per block are limited resources partitioned among the active warps.
- **Occupancy**: Ratio of active warps to the maximum possible warps that can be executed on an SM. High occupancy leads to better performance.
  - **Factors affecting Occupancy**:
    - Number of registers per thread.
    - Shared memory usage per block.
    - Thread count per block.

## 4.8 Querying Device Properties

- **CUDA API Functions**:
  - `cudaGetDeviceProperties()`: Queries properties such as memory, clock rate, and maximum threads per block.
  - **Key Properties**:
    - Number of SMs, maximum thread count per block, and warp size.
    - Shared memory, register allocation, and constant memory.

## 4.9 Summary

- **Modern GPUs** are highly parallel processors that rely on efficient scheduling of blocks and warps.
- **Warps and SIMD execution** enable massive parallelism, but divergence and resource partitioning can affect performance.
- **Occupancy and latency tolerance** are crucial for maximizing GPU performance.
- **Transparent scalability** allows CUDA applications to adapt to various GPU architectures by balancing resources and managing thread synchronization.

---
