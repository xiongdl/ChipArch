# CA4AI
Awesome Materials on Topic "Computer Architecture for AI". 

# Book
- 2024 Springer - Towards Heterogeneous Multi-core Systems-on-Chip for Edge Machine Learning.
  - Website: https://drive.google.com/file/d/1erxYYDDZk4vAGDIqe1unNvaAXiFi5r6L/view
- 2024 - Machine Learning System with TinyML. (Harvard University, CS249r)
  - Website: https://harvard-edge.github.io/cs249r_book/Machine-Learning-Systems.pdf
- 2020 SLCA - Efficient Processing of Deep Neural Networks.
- 2020 SLCA - Data Orchestration in Deep Learning Accelerators.

# Course
- 2024 Spring - Hardware for Machine Learning. (Berkeley EE290-2)
  - Website: https://inst.eecs.berkeley.edu/~ee290-2/sp24/

# Tutorial
- 2019 ISCA - Hardware Architectures for Deep Neural Networks. (MIT, NVIDIA)
  - Website: http://eyeriss.mit.edu/tutorial.html

# Insight
- 2024 ISSCC - Semiconductor Industry: Present & Future. (TSMC)
  - Takeaways:
    - Continued advanced technology scaling: new device architecture (CFET), low dimensional channel materials.
    - Essential design-technology co-optimization (DTCO): extract maximum values by tailoring technology definition (standard logic cell, SRAM, etc).
    - Essential system-technology co-optimization (STCO): logic integration (2.5D + 3D integration), memory bandwidth (memory + logic), specially for power delivery (voltage regulator integration), specially for interconnect speed (OE on substrate).
- 2024 ISSCC - Computing in the Era of Generative AI. (NVIDIA)
  - Takeaways:
    - Being an early adopter is hard and uncomfortable but I encourge you to believe it can pay off.
    - The most important lesson is not expect immediate miracles. You have to be persistent, especially if you're trying a new era.
    - Instead, believe that AI will fundamentally transform the semiconductor industry and your business.
  

# Survey
- Commercial AI Accelerators Survey from MIT Lincoln Laboratory Supercomputing Center.
  - 2023 HPEC - Lincoln AI Computing Survey (LAICS) Update.
  - 2022 HPEC - AI and ML Accelerator Survey and Trends.
  - 2021 HPEC - AI Accelerator Survey and Trends.
  - 2020 HPEC - Survey of Machine Learning Accelerators.
  - 2019 HPEC - Survey and Benchmarking of Machine Learning Accelerators.
- 2022 TCSI - Reconfigurability, Why It Matters in AI Tasks Processing: A Survey of Reconfigurable AI Chips. (Tsinghua University)
- 2020 JPROC - Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey. (Tsinghua University)
- 2017 JPROC - Efficient Processing of Deep Neural Networks: A Tutorial and Survey. (MIT)
- Neural Network Accelerator Comparison from NICS EFC Lab of Tsinghua University.
  - Website: https://nicsefc.ee.tsinghua.edu.cn/projects/neural-network-accelerator/
- Computation Used to Train Notable Artificial Intelligence Systems
  - Website: https://ourworldindata.org/grapher/artificial-intelligence-training-computation 

# Paper Reading
- The review questions guide you through the paper reading process. (cite from EE290-2 slides)
  - What are the **motivations** for this work?
  - What is the **proposed solution**?
  - What is the work's **evaluation** of the proposed solution?
  - What is your **analysis** of the identified problem, idea and evaluation?
  - What are **future directions** for this research?
  - What **questions** are you left with?
  
# Conference
### 2024 ISSCC
- ATOMUS: A 5nm 32TFLOPS/128TOPS ML System-on-Chip for Latency Critical Applications. (Rebellions)
  - Direction: Hardware (Industrial Product) for Low Latency
  - Keywords: Multi-Level Synchronization, Latency-Centric Memory
  - Takeaways:
    - ATOMUS achieves high inference capability with outstanding single-stream responsiveness for demanding service-layer objective (SLO)-based AI services and pipelined inference applications, including large language models (LLM).
    - Latency criticality in AI-as-a-Service is gaining more attention where batching is not always preferable, especially for a long pipelined service stack and tight SLO services stacks.
    - ATOMUS comprises flexible AI compute cores and a data/control on-chip network in conjunction with latency-centric memory architectures.
- AMD Instinct MI300 Series Modular Chiplet Package - HPC and AI Accelerator for Exa-Class Systems.(AMD)
  - Direction: Hardware (Industrial Product) for High Performance
  - Keywords: Chiplet, Advanced Packing
  - Takeaways:
    - Two new chiplet types are introduced in MI300, the input/output die (IOD) and the accelerator complex die (XCD).
    - MI300 is not just the first AMD multiple die hybrid bonded architecture but also the first AMD hybrid bonded 3D+2.5D Architecture.
- IBM NorthPole: An Architecture for Neural Network Inference with a 12nm Chip. (IBM)
  - Direction: Hardware (Industrial Product) for Low Latency
  - Keywords: Mix-Precision
  - Takeaways:
    - The NorthPole Inferece Chip comprises a 256-Core Array with 192MB of distributed SRAM, a Frame Buffer memory with 32MB of SRAM, and an I/O interface. At nominal 400MHz frequency, cores deliver a cumulative peak compute performance of over 200 TOPS at 8b-, 400 TOPS at 4b-, and 800 TOPS at 2b-precision with high utilization.
    - NorthPole supports layer-specific precision selection, so some layers can be 8b, while others are 4b and/or 2b.
    - All communication, computation, control and memory access operations are fully deterministic and stall-free, timing is scheduled by the compiler.
- Metis AIPU: A 12nm 15TOPS/W 209.6TOPS SoC for Cost- and Energy-Efficient Inference at the Edge. (Axelera AI)
  - Direction: Hardware (Industrial Product) for Cost- and Energy-Efficient
  - Keywords: Digital In-Memory Computing
  - Takeaways:
    - Metis leverages the benefits from a quantized digitial in-memory computing (D-IMC) architecture - with 8b weights, 8b activations, and full-precision accumulation - to descrease both the memory cost of weights and activations and the energy consumption of matrix-vector multiplications (MVM), without comprising the neural network accuracy.
    - The primary component of the Metis AIPU is the AI Core, comprising the D-IMC for MVM operations, a data processing unit (DPU) for element-wise vector operations and activations, a depth-wise processing unit (DWPU) for depth-wise convolution, pooling, and up-sampling, a local 4MiB L1 SRAM, and a RISC-V control core.
    - The performance of the AIPU is a result of a vast digital in-memory compute array, signficantly larger than previous designs, coupled with gating techniques ensuring top-tier energy efficiency even in low utilization.
- **A 23.9TOPS/W @ 0.8V, 130TOPS AI Accelerator with 16x Performance-Accelerable Pruning in 14nm Heterogeneous Embedded MPU for Real-Time Robot Applications**. (Renesas Electronics)
  - Direction: Hardware (Industrial Product) for Power-Efficient
  - Keywords: Pruning, Dynamically Reconfigurable Processor, Robot
  - Takeaways:
    - We propose a power-efficient AI-MPU including: 1) a flexible (N:M) pruning rate control technology capable of up to 16x AI performance acceleration, and 2) a heterogenous architecture for multi-task & real-time robot operation based on the co-operation among a dynamically reconfigurable processor (DRP), AI accelerator (DRP-AI) and embedded CPU.
    - We developed a flexible N:M pruning method that can greatly relax pruning position constraints from structured pruning, while maintaining the ability to process weights in parallel computing units.
    - For CNN processing, computationally dominant convolution layers and common layers are handled by the MAC. By using the DRP, the inference speed, including pre- and post- processing, is 6.5x faster than using the embedded CPU.

# Transaction

