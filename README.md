# CA4AI
Awesome Materials on Topic "Computer Architecture for AI". 

# Table of Contents
- [Book](#book)
- [Course](#course)
- [Tutorial](#tutorial)
- [Survey](#survey)
- [How to Read a Paper?](#how-to-read-a-paper)
- [Direction](#direction)
  - [Analytical Framework](#analytical-framework)
  - [LLM Accelerator](#llm-accelerator)
  - [CNN Accelerator](#cnn-accelerator)
  - [In-Memory Computing](#in-memory-computing)
- [Paper List](#paper-list)
  - 2024: [ASPLOS](#2024-asplos), [ISSCC](#2024-isscc)
  - 2017: [JSSC](#2017-jssc)
  - 2016: [ISCA](#2016-isca), [ISSCC](#2016-isscc)

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
- 2024 ISSCC Short Course - Machine Learning Hardware: Considerations and Accelerator Approaches.

# Tutorial
- 2019 ISCA - Hardware Architectures for Deep Neural Networks. (MIT, NVIDIA)
  - Website: http://eyeriss.mit.edu/tutorial.html
- 2020 ISPASS - Timeloop/Accelergy Tutorial: Tools for Evaluating Deep Neural Network Accelerator Designs. (MIT, NVIDIA)
  - Website: https://accelergy.mit.edu/tutorial.html
- 2020 MICRO - MAESTRO Tutorial
  - Website: https://maestro.ece.gatech.edu/docs/build/html/tutorials.html
- 2021 ISCA - Sparse Tensor Accelerators: Abstraction and Modeling. (MIT, NVIDIA)
  - Website: https://accelergy.mit.edu/sparse_tutorial.html
 

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

# How to Read a Paper?
- The review questions guide you through the paper reading process. (cite from EE290-2 slides)
  - What are the **motivations** for this work?
  - What is the **proposed solution**?
  - What is the work's **evaluation** of the proposed solution?
  - What is your **analysis** of the identified problem, idea and evaluation?
  - What are **future directions** for this research?
  - What **questions** are you left with?

# Direction
## Analytical Framework

## LLM Accelerator

## CNN Accelerator

## In-Memory Computing


  
# Paper List
## 2024 ASPLOS
### AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference. (Seoul National University) <a name="attacc@2024_asplos"></a>
- Direction: Efficient LLM Inference
- Keywords: In Memory Computing
- Takeaways:
  - Identify the growing importance of the attention layer in the trend with increasing model sizes and sequence length.
  - The conventional serving platforms, such as GPUs, are suboptimal for large batch sizes having stringent memory capacity and bandwidth requirements in processing the attention layer under a tight service-level objective (SLO).
  - AttAcc: a DRAM-based processing-in-memory (PIM) architecture to accelerate memory-bound attention layers; increase the maximum batch size under the SLO constraint.
  - A heterogeneous system architecture: strategically processes the memory-bound attention layers with AttAcc, while efficiently handling compute-bound batched FC layers with xPUs.


### 8-bit Transformer Inference and Fine-Tuning for Edge Accelerators. (Stanford University)
- Direction: Efficient LLM Inference
- Keywords: Quantization, Fine-Tuning, Softmax
- Takeaways:
  - This paper is the first to systematically explore quantization of all Transformer operations beyond GEMM via FP8 and Posit8.
  - An area- and power-efficient posit softmax is designed to compensate for the larger posit MAC unit.
  - Operaton fusion is employed to reduce the post-training quantization accuracy loss, simultaneously enhancing the fine-tuning accuracy.

### Carat: Unlocking Value-Level Parallelism for Multiplier-Free GEMMs. (University of Wisconsin-Madison)
- Direction: Efficient LLM Inference
- Keywords: Value Reuse, GEMM
- Takeaways:
  - Value-Level Parallism: unique products are computed only once, and different input subscribe to (select) their products via temporal coding.

### Atalanta: A Bit is Worth a "Thousand" Tensor Values. (University of Toronto)
- Keywords: Lossloss Tensor Compression
- Takeaways:
  - Atalanta is a practical and lossless tensor compression method. It enables transparent and highly-efficient encoding for weights and activations, it is low-cost and can be seamlessly integrated with SoTA deep learning accelerators.

### SpecInfer: Accelerating Large Language Model Serving with Tree-Based Speculative Inference and Verification. (CMU)
- Direction: Efficient LLM Serving
- Keywords: Speculative Inference
- Takeaways:
  - A key insight behind SpecInfer is to simultaneously consider a diversity of speculation candidates to efficiently predict the LLM's outputs, which are organized as a token tree and verified against the LLM in parallel using a tree-based parallel decoding mechanism. 

## 2024 ISSCC
### Forum: Eenergy-Efficient AI-Computing Systems for Large-Language Models.
- LLM Training and Inference on GPU and HPC Systems.
- LLM Energy Problem.
- Quantizing LLMs for Efficient Inference at the Edge.
- Next-Generation Mobile Processors with Large-Lanuage Models and Large Multimodal Models.

### ATOMUS: A 5nm 32TFLOPS/128TOPS ML System-on-Chip for Latency Critical Applications. (Rebellions)
- Direction: Low-Latency Chip
- Keywords: Multi-Level Synchronization, Latency-Centric Memory
- Takeaways:
  - ATOMUS achieves high inference capability with outstanding single-stream responsiveness for demanding service-layer objective (SLO)-based AI services and pipelined inference applications, including large language models (LLM).
  - Latency criticality in AI-as-a-Service is gaining more attention where batching is not always preferable, especially for a long pipelined service stack and tight SLO services stacks.
  - ATOMUS comprises flexible AI compute cores and a data/control on-chip network in conjunction with latency-centric memory architectures.
      
### AMD Instinct MI300 Series Modular Chiplet Package - HPC and AI Accelerator for Exa-Class Systems.(AMD)
- Direction: Chip (Industrial Product) for High Performance
- Keywords: Chiplet, Advanced Packing
- Takeaways:
  - Two new chiplet types are introduced in MI300, the input/output die (IOD) and the accelerator complex die (XCD).
  - MI300 is not just the first AMD multiple die hybrid bonded architecture but also the first AMD hybrid bonded 3D+2.5D Architecture.
      
### IBM NorthPole: An Architecture for Neural Network Inference with a 12nm Chip. (IBM)
- Direction: Low-Latency Chip
- Keywords: Mix-Precision
- Takeaways:
  - The NorthPole Inferece Chip comprises a 256-Core Array with 192MB of distributed SRAM, a Frame Buffer memory with 32MB of SRAM, and an I/O interface. At nominal 400MHz frequency, cores deliver a cumulative peak compute performance of over 200 TOPS at 8b-, 400 TOPS at 4b-, and 800 TOPS at 2b-precision with high utilization.
  - NorthPole supports layer-specific precision selection, so some layers can be 8b, while others are 4b and/or 2b.
  - All communication, computation, control and memory access operations are fully deterministic and stall-free, timing is scheduled by the compiler.
      
### Metis AIPU: A 12nm 15TOPS/W 209.6TOPS SoC for Cost- and Energy-Efficient Inference at the Edge. (Axelera AI)
- Direction: Cost- and Energy-Efficient Chip
- Keywords: Digital In-Memory Computing
- Takeaways:
  - Metis leverages the benefits from a quantized digitial in-memory computing (D-IMC) architecture - with 8b weights, 8b activations, and full-precision accumulation - to descrease both the memory cost of weights and activations and the energy consumption of matrix-vector multiplications (MVM), without comprising the neural network accuracy.
  - The primary component of the Metis AIPU is the AI Core, comprising the D-IMC for MVM operations, a data processing unit (DPU) for element-wise vector operations and activations, a depth-wise processing unit (DWPU) for depth-wise convolution, pooling, and up-sampling, a local 4MiB L1 SRAM, and a RISC-V control core.
  - The performance of the AIPU is a result of a vast digital in-memory compute array, signficantly larger than previous designs, coupled with gating techniques ensuring top-tier energy efficiency even in low utilization.
    
### A 23.9TOPS/W @ 0.8V, 130TOPS AI Accelerator with 16x Performance-Accelerable Pruning in 14nm Heterogeneous Embedded MPU for Real-Time Robot Applications. (Renesas Electronics)
- Direction: Power-Efficient Chip
- Keywords: Pruning, Dynamically Reconfigurable Processor, Robot
- Takeaways:
  - We propose a power-efficient AI-MPU including: 1) a flexible (N:M) pruning rate control technology capable of up to 16x AI performance acceleration, and 2) a heterogenous architecture for multi-task & real-time robot operation based on the co-operation among a dynamically reconfigurable processor (DRP), AI accelerator (DRP-AI) and embedded CPU.
  - We developed a flexible N:M pruning method that can greatly relax pruning position constraints from structured pruning, while maintaining the ability to process weights in parallel computing units.
  - For CNN processing, computationally dominant convolution layers and common layers are handled by the MAC. By using the DRP, the inference speed, including pre- and post- processing, is 6.5x faster than using the embedded CPU.
      
### C-Transformer: A 2.6-18.1uJ/Token Homogeneous DNN-Transformer/Spiking-Transformer Processor with Big-Little network and Implicit Weight Generation for Large Language Models. (KAIST)
- Direction: Power-Efficient and Low-Latency Chip
- Keywords: Big-Little Network, Implicit Weight Generation, Transformer
- Takeaways:
  - Three solutions for large External Memory Access (EMA) overhead of LLM: 1) big-little network; 2) implicit weight generation; 3) extended sign compression.
    
### NVE: A 3nm 23.2TOPS/W 12b-Digital-CIM-Based Neural Engine for High-Resolution Visual-Quality Enhancement on Smart Devices. (MediaTek & TSMC)
- Direction: Energy-Efficient Chip
- Keywords: Super-Resolution, Noise-Reduction, Digital In-Memory Computing, Fusion
- Takeaways:
  - Neural Visual Enhancement Engine (NVE) has 3 features: 1) a weight-reload-free Digital-Compute-In-Memory (DCIM) engine with reduced weight switching rate to enhance the computational power efficiency; 2) a Convolutional Element (CE) fusion establishes a workload-balanced pipeline architecture, reducing external memory access and power consumption; 3) an adaptive data control and striping optimization mechanism supports stride convolution and transposed convolution in DCIM with improved utilization, and an optimized execution flow for efficient data traversal.
    
### Vecim: A 289.13GOPS/W RISC-V Vector Co-Processor with Compute-in-Memory Vector Register File for Efficient High-Performance Computing. (University of Texas)
- Direction: Area- and Energy-Efficent Chip
- Keywords: RISC-V, Digital In-Memory Computing, Vector Processor
- Takeaways:
  - Vecim presents Compute-in-Memory (CIM) as a microarchitecture technique to address the limitations: 1) the expensive on-chip data movement; 2) the high off-chip memory bandwidth requirement; and 3) the Vector Register File (VRF) complexity.
  - A 1R1W SRAM VRF supporting all-digital in-memory INT8/BF16/FP16 multiplication and addition is first presented, eliminating the data movement between the VRF and ALU/FPU without increasing the complexity of the VRF organization.

### A Software-Assisted Peak Current Regulation Scheme to Improve Power-Limited Inference Performance in a 5nm AI SoC. (IBM)
- Direction: Power-Efficient Chip
- Keywords: Power Mangement
- Takeaways:
  - This work leverages the unique characteristic of AI workloads, which allows predictive compile-time software optimization and proposes a new power management architecture to minimize worst-case margins and realize the potential of AI accelerators.
  - A new software-assisted feed-forward current-limiting scheme is proposed in conjunction with PCIe-card-level closed-loop control to maximize performance under sub-ms peak current constraints.

## 2017 JSSC
### Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks. (MIT) <a name="eyeriss@2017_jssc"></a>
- Direction: Energy-Efficent Chip
- Keywords: Row-Stationary Dataflow, Run Length Encoding, Data Gating
- Takeaways:
  - A spatial architecture using an array of 168 processing elements (PEs) that creates a four-level memory hierarchy.
  - The row-stationary dataflow minimizes data movement for all data types (ifmap, filter, and psums/ofmap) simultaneously.
  - A network-on-chip (NoC) architecture that uses both multicast and point-to-point single-cycle data delivery to support the RS dataflow.
  - Run-length compression (RLC) and PE data gating that exploit the statistics of zero data in CNNs to further improve energy efficiency.
  - Even though the 168 PEs are identical and run under the same core clock, their processing states do no need to proceed in lock steps, i.e., not as a systolic array. Each PE can start its own processing as soon as any fmaps or psums arrives.

## 2016 ISCA
### Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks. (MIT) <a name="eyeriss@2016_isca"></a>
- Direction: Energy-Efficient Dataflow
- Keywords: Row-Stationary Dataflow, Analysis Framework
- Takeaways:
  - An analysis framework that can quantify the energy efficiency of different CNN dataflows (IS, OS, WS, NLR, RS) under fixed area and processing parallelism constraints.
  - Compared with SIMD/SIMT architectures, spatial architectures are particularly suitable for applications whose dataflow exhibits producer-consumer relationships or can leverage efficient data sharing among a region of PEs.
  - DRAM bandwidth alone does not dictate energy-efficiency; dataflows that require high bandwidth to the on-chip global buffer can also result in significant energy cost.
  - For all dataflows, increasing the size of PE array helps to improve the processing throughput at similar or better energy efficiency.
  - Larger batch sizes also result in better energy efficiency in all dataflows, except for WS, which suffers from insufficient global buffer size.


## 2016 ISSCC
### Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks. (MIT) <a name="eyeriss@2016_isscc"></a>
- See [Eyeriss @ 2017 JSCC](#eyeriss@2017_jssc)

