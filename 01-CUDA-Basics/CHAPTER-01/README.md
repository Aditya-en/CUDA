# Chapter 1: Introduction to Massively Parallel Processing

## 1. The Evolution of Computing and Parallel Processing

### Historical Context

- Early applications relied on advancements in processor speed, memory speed, and capacity
- Single-CPU microprocessors drove performance increases in the 1980s and 1990s
- Progress slowed in 2003 due to energy consumption and heat dissipation issues

### The Shift to Parallel Computing

- Industry moved from single-core to multi-core and many-thread processors
- Two main trajectories emerged:
  1. **Multicore**:
     - Aims to maintain sequential program speed while increasing core count
     - Example: Intel multicore server microprocessor with up to 24 cores
  2. **Many-thread**:
     - Focuses on execution throughput of parallel applications
     - Example: NVIDIA Tesla A100 GPU with tens of thousands of threads

### Performance Comparison

- GPU vs CPU peak floating-point performance (as of 2021):
  - NVIDIA Tesla A100 GPU:
    - 9.7 TFLOPS for 64-bit double-precision
    - 156 TFLOPS for 32-bit single-precision
    - 312 TFLOPS for 16-bit half-precision
  - Recent Intel 24-core CPU:
    - 0.33 TLOPS for double-precision
    - 0.66 TFLOPS for single-precision

## 2. GPU vs CPU Design Philosophy

### CPU (Latency-Oriented Design)

- Optimized for sequential code performance
- Features:
  - Large last-level on-chip caches
  - Sophisticated branch prediction and execution control logic
  - Designed to minimize effective latency of operations
- Drawbacks:
  - Consumes more chip area and power per arithmetic unit

### GPU (Throughput-Oriented Design)

- Shaped by demands of video game industry
- Features:
  - Maximizes chip area and power budget for floating-point calculations and memory access
  - Operates at ~10 times the memory bandwidth of contemporary CPUs
  - Designed for massive parallelism in accessing memory
- Advantages:
  - Higher peak floating-point performance
  - Greater memory bandwidth

### Key Differences

- Reducing latency is more expensive than increasing throughput in terms of power and chip area
- GPUs optimize for execution throughput of massive numbers of threads
- GPUs allow for longer latency in memory channels and arithmetic operations to save chip area and power

## 3. Evolution of GPU Computing

### Early Stages: GPGPU

- Graphics chips were difficult to use for general-purpose computing
- Programmers had to use graphics API functions (OpenGL or Direct3D)
- Limited the types of applications that could be developed

### CUDA Revolution (2007)

- NVIDIA added hardware support for general-purpose computing
- Introduced a new programming interface bypassing graphics APIs
- Provided familiar C/C++ programming tools
- Greatly expanded the types of applications that could be developed for GPUs

### Impact on the Market

- GPUs have been sold by hundreds of millions
- Over 1 billion CUDA-enabled GPUs in use
- Made GPUs economically attractive targets for application developers

## 4. Motivations for Increased Computing Speed

### Scientific Applications

- Molecular biology simulations with computational models
- Weather forecasting timeliness and accuracy

### Media and Entertainment

- High-definition video processing and manipulation
- 3D imaging and visualization
- Realistic physics simulation in gaming

### User Interfaces

- Natural interfaces with high-resolution touch screens
- 3D perspectives, virtual and physical space information integration
- Voice and computer vision-based interfaces

### Artificial Intelligence

- Deep learning based on artificial neural networks
- Enabled by increased data availability and GPU computing power

### Industry Applications

- Digital twins for stress testing and deterioration prediction

## 5. Speedup and Amdahl's Law

### Defining Speedup

- Speedup = Execution time on old system / Execution time on new system

### Amdahl's Law

- Speedup limited by the non-parallelizable portion of the code
- Formula: Speedup = 1 / ((1 - P) + (P / S))
  - P: Proportion of execution time that can be parallelized
  - S: Speedup of parallelized portion

### Examples

1. If 30% can be parallelized:
   - Even with infinite speedup of parallel portion, max overall speedup is 1.43x
2. If 99% can be parallelized:
   - 100x speedup in parallel portion yields 50x overall speedup

### Practical Considerations

- Achieving 100x+ speedup requires extensive optimization
- Memory bandwidth often limits initial parallelization attempts to ~10x speedup
- Optimization techniques crucial for overcoming hardware limitations

## 6. Challenges in Parallel Programming

### Algorithm Design

- Designing parallel algorithms with same computational complexity as sequential ones
- Some parallel algorithms may do more total work than sequential counterparts

### Memory Management

- Many applications are memory-bound
- Techniques needed for improving memory access speed and efficiency

### Data Characteristics Sensitivity

- Parallel program performance often more sensitive to input data characteristics
- Challenges with erratic data sizes and uneven distributions

### Synchronization Overhead

- Some applications require thread collaboration
- Synchronization operations (barriers, atomic operations) can impose significant overhead

### Strategies for Addressing Challenges

- Use of algorithm primitives like prefix sum
- Techniques for improving memory access patterns
- Methods for regularizing data distributions
- Strategies for reducing synchronization overhead

## 7. Related Parallel Programming Interfaces

### OpenMP

- For shared memory multiprocessor systems
- Uses compiler directives and pragmas
- Advantages: Automation and abstraction for better portability
- Still requires understanding of parallel programming concepts

### MPI (Message Passing Interface)

- For scalable cluster computing
- All data sharing done through explicit message passing
- Widely used in High-Performance Computing (HPC)
- Requires significant effort for domain decomposition and data exchange management

### OpenCL

- Standardized programming model developed by industry players
- Similar to CUDA but relies more on APIs than language extensions
- Offers portability across different vendors' processors

## 8. CUDA Advantages

- Explicit control over parallel programming details
- Excellent learning vehicle for parallel programming concepts
- Provides shared memory model within GPU
- Easier to achieve high performance compared to some alternatives
- Extensive tool support for debugging and performance optimization

## 9. Overarching Goals of the Book

1. Teach high-performance parallel programming

   - Focus on techniques for developing high-performance parallel code
   - Emphasis on computational thinking for massively parallel processors

2. Ensure correct functionality and reliability

   - Address challenges in debugging and supporting parallel code
   - Focus on data parallelism for both performance and reliability

3. Develop scalable code for future hardware generations
   - Techniques for regularizing and localizing memory data accesses
   - Ensuring applications can scale with new generations of parallel hardware

## 10. Importance of Hardware Understanding

- Knowledge of hardware architecture crucial for effective parallel programming
- Book will cover GPU architecture fundamentals in Chapter 4
- Specialized architecture concepts discussed alongside programming techniques

## 11. Key Concepts Introduced

- Heterogeneous computing: Utilizing both CPUs and GPUs effectively
- Throughput vs Latency oriented design in processor architecture
- Amdahl's Law and its implications for parallel speedup
- Memory-bound vs Compute-bound applications
- Work efficiency in parallel algorithms

## 12. Looking Ahead

- Coverage of parallel programming principles and patterns
- Focus on practical application of techniques in real-world scenarios
- Exploration of important parallel computation patterns and applications
- Discussion on programming for heterogeneous computing clusters
