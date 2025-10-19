# Towards Efficient RAG: Quantization-Aware Adaptation of LongRefiner

## 2. Team Members (2)  
- Junzhi Chen
- Chun-Ju Tao

## 3. Goal / Objective  
This project aims to investigate how different quantization strategies affect the efficiency and performance of large language model (LLM) fine-tuning in Long-context Retrieval-augmented Generation tasks.  
We focus on two key modules from LongRefiner (Jin et al., 2025)—Dual-Level Query Analysis (DQA) and Adaptive Document Refinement (ADR)—which are critical for long-context retrieval-augmented generation (RAG).  

Specifically, we compare three training paradigms:
1. Full LoRA (FP16) — standard fine-tuning with full-precision weights.  
2. LoRA + PTQ — full-precision training followed by post-training quantization (int8/int4).  
3. QLoRA — quantization-aware fine-tuning using 4-bit NF4 quantization during training.  

The goal is to determine whether quantization-aware fine-tuning can achieve better trade-offs between accuracy, training efficiency, and deployment cost.

## 4. Challenges
1. Balancing accuracy and efficiency: Quantization reduces memory usage but introduces representational noise that may degrade DQA classification accuracy and ADR ranking stability.  
2. Ensuring fair comparison: Each approach must share identical data, LoRA configurations, and optimization schedules to isolate the effect of quantization strategy.  
3. Data and supervision consistency: The teacher model (3B LongRefiner) provides outputs for distillation; aligning student predictions (especially under quantized noise) with teacher logits is non-trivial.  
4. Quantization calibration and stability: Post-training quantization depends on careful calibration, while QLoRA may slow early-stage convergence due to quantization-aware noise.

## 5. Approach / Techniques

### Overall Design
We fix the Hierarchical Document Structuring (HDS) component using the original 3B LongRefiner-LoRA model (full precision), and train lighter student models for DQA and ADR.  
The study compares three versions of student fine-tuning pipelines under identical settings.

| Group | Training Mode | Quantization Stage | Backbone | Adapter Precision |
|--------|----------------|--------------------|-----------|--------------------|
| A. Full LoRA (Baseline) | FP16 training | None | Qwen 0.5B | FP16 |
| B. LoRA + PTQ | FP16 training | Post-training (int8/int4) | Qwen 0.5B | FP16 |
| C. QLoRA | 4-bit NF4 training (quantization-aware) | During training | Qwen 0.5B | FP16 |

### Dual-Level Query Analysis (DQA)
- **Input:** Query text  
- **Output:** Local/global relevance weights ($$R_q = \text{Softmax}(P_l, P_g)$$)  
- **Objective:** Minimize cross-entropy and KL divergence between teacher and student logits.  

### Adaptive Document Refinement (ADR)
- **Input:** Query + document outline (titles & abstracts)  
- **Output:** Section/node relevance scores  
- **Objective:** Distill teacher scores via KL loss or pairwise ranking loss (ListNet).  

## 6. Implementation Details

| Component | Description |
|------------|-------------|
| **Hardware** | NVIDIA A100 (baseline) and RTX 4090 / RTX 3060 (edge simulation) |
| **Compute Type** | Local GPU nodes in NYU HPC;|
| **Software** | PyTorch, Hugging Face Transformers, PEFT, bitsandbytes |
| **Existing Codebase** | [LongRefiner GitHub repository](https://github.com/ignorejjj/LongRefiner) for teacher model inference code, [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training code and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) for dataset and inference|
| **Dataset** | The training dataset is constructed using the version collected by FlashRAG(dataset detail can be seen in the evaluation part). We use the first 10,000 samples from the training set of each dataset, which are merged to form the final dataset.|

## 7. Evaluation Details

### **a. Evaluation Overview**
The evaluation is designed to assess both **task performance** and **system efficiency** of the three fine-tuning paradigms:
- **Full LoRA (FP16)**
- **LoRA + PTQ (Post-Training Quantization)**
- **QLoRA (Quantization-Aware Fine-Tuning)**

We will reproduce the **open-domain QA benchmark results** similar to Table 2 in *LongRefiner (Jin et al., 2025)*, while extending the evaluation with detailed **training and inference metrics** to capture system-level efficiency.

---

### **b. Benchmark Datasets**
We evaluate on **seven open-domain QA datasets** covering single-hop, multi-hop, and long-form question answering:

| Type | Dataset | Description |
|------|----------|--------------|
| Single-hop | NQ, TriviaQA, PopQA | Fact-based QA tasks |
| Multi-hop | HotpotQA, 2Wiki | Require multi-step reasoning |
| Long-form | ASQA, ELI5  | Require extended contextual retrieval and synthesis |

---

### **c. Task-Level Metrics**
We report **accuracy (Acc)** and **F1 score** for each dataset to evaluate reasoning quality and answer overlap, consistent with the LongRefiner benchmark.  

| Metric | Meaning |
|---------|----------|
| **Acc** | Exact match rate of predicted answers |
| **F1** | Token-level harmonic mean of precision and recall |
| **Avg. F1** | Averaged across all datasets as overall QA performance |

---

### **d. System-Level Metrics**
In addition to task performance, we evaluate both **training** and **inference** efficiency:

#### **Training Metrics**
| Metric | Description |
|---------|--------------|
| **Training VRAM (GB)** | Peak GPU memory consumption during fine-tuning |
| **Throughput (samples/s)** | Number of processed samples per second |
| **Training Time per Epoch (min)** | Total time for one full epoch |
| **Total Training FLOPs (TFLOPs)** | Estimated total compute used for convergence, derived as `FLOPs/step × #steps × #GPUs` |

#### **Inference Metrics**
| Metric | Description |
|---------|--------------|
| **Inference VRAM (GB)** | Memory usage during model inference |
| **Latency (ms/sample)** | End-to-end inference time per query |
| **FLOPs per Query (GFLOPs)** | Estimated number of operations required to process a single inference request |
| **Model Size (GB)** | Disk size of quantized vs. full-precision checkpoints |

These metrics will quantify the trade-offs between computational efficiency and model quality under different quantization strategies.

---

### **e. Comparative Analysis**
Each model variant will be evaluated across all QA datasets and system metrics.  
The final analysis will include:

1. **Performance Retention:**  
   F1 and Accuracy difference between Full LoRA and QLoRA/LoRA+PTQ.

2. **Efficiency Gains:**  
   Relative reduction in training VRAM, inference latency, and FLOPs.

3. **Pareto Frontier Visualization:**  
   Plot *Accuracy (F1)* vs. *Efficiency (VRAM/Latency/FLOPs)* to highlight optimal trade-offs.

4. **Convergence Curves:**  
   Compare validation loss and F1 progression across training steps to assess QLoRA stability.

---

### **f. Expected Outcomes**
- **QLoRA** is expected to retain a comparable accuracy with full-precision accuracy while reducing both training and inference memory.  
- **LoRA + PTQ** may exhibit higher quantization error under int4 settings but similar inference efficiency.  
- **Full LoRA** will serve as the upper performance bound but with significantly higher computational cost.

This comprehensive evaluation provides both *task-level* and *system-level* evidence for selecting efficient fine-tuning strategies for retrieval-augmented generation (RAG) systems.

## 8. Demo Planned
We will develop an interactive demo comparing the three fine-tuned models on a retrieval-augmented QA task.  
Users can input a question, and the system will display:
- DQA classification (local/global ratio)  
- ADR-selected document nodes  
- Final retrieved context and generated QA answer  

The demo will visualize inference latency, VRAM usage, and answer quality side by side for each variant (Full LoRA, LoRA+PTQ, QLoRA).

## 9. References

**[1] Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yongkang Wu, Zhonghua Li, Ye Qi, and Zhicheng Dou. 2025. Hierarchical Document Refinement for Long-context Retrieval-augmented Generation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3502–3520, Vienna, Austria. Association for Computational Linguistics.** 

**Contribution:** Introduced LongRefiner, a multi-stage framework integrating Hierarchical Document Structuring (HDS), Dual-Level Query Analysis (DQA), and Adaptive Document Refinement (ADR). It achieved superior QA performance by modeling document hierarchy and query granularity jointly.  

**[2] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36, 10088-10115.**  

**Contribution:** Proposed quantization-aware LoRA fine-tuning using 4-bit NF4 quantization, achieving near full-precision performance while reducing memory consumption by over 60%.  

**[3] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. ICLR, 1(2), 3.**  

**Contribution:** Introduced LoRA, a parameter-efficient method for fine-tuning large language models by injecting low-rank adapters, allowing rapid adaptation with minimal training cost.  

**How Our Work Builds on These:**  
Our project combines the parameter-efficient LoRA framework [3] and quantization-aware optimization from QLoRA [2] into the structured reasoning framework of LongRefiner [1].  
While previous studies applied QLoRA mainly to generic text generation, we evaluate its impact on structured, multi-stage RAG tasks.  
By systematically comparing full-precision, post-training quantized, and quantization-aware fine-tuning setups, we aim to provide insights into how quantization affects reasoning stability and retrieval accuracy within the LongRefiner pipeline.
