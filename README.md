# HPML_Final_Project

Team:<br>
| | |
|---|---|
| Hechuan Liang | hl5035 |
| Yuyang Ji | yj2669 |

## Project Description
The primary objective of this project is to improve the efficiency and scalability of the Mixture of Experts (MoE) model by employing a random routing method inspired by the AdaMix: Mixture-of-Adaptations approach. This will aim to increase the performance without adding to the computational cost.

## Milestones

### 1. Conceptualization and Research
- **Objective:** Understand and conceptualize the Mixture of Experts (MoE) framework and its integration with Transformer models.
- **Activities:** Literature review, studying existing MoE models, and understanding Transformer architecture.
- **Status:** [Completed]

### 2. Design and Development of Random Routing Algorithm
- **Objective:** Develop an algorithm for random routing to select Multi-Layer Perceptrons (MLPs) efficiently.
- **Activities:** Algorithm design, coding, and initial testing on a small scale.
- **Status:** [Completed]

### 3. Integration of Random Routing with Transformer Model
- **Objective:** Seamlessly integrate the developed random routing algorithm into a Transformer model.
- **Activities:** Code merging, integration testing, and troubleshooting.
- **Status:** [Completed]

### 4. Comprehensive Testing and Validation
- **Objective:** Test the integrated model to ensure functionality and efficiency.
- **Activities:** Deploying test cases, analyzing model performance, and making necessary adjustments.
- **Status:** [Completed]

### 5. Averaging MLP Weights During Testing
- **Objective:** Implement and test the averaging of weights of each MLP during the testing phase to enhance model performance.
- **Activities:** Developing the methodology for weight averaging, applying it during tests, and evaluating its impact on the overall model performance.
- **Status:** [Completed]

### 6. Performance Analysis and Optimization
- **Objective:** Analyze the results to identify performance bottlenecks and areas for improvement.
- **Activities:** Performance metrics analysis, bottleneck identification, and model optimization.
- **Status:** [Completed]

### 7. Documentation and Reporting
- **Objective:** Document the process, findings, and final model specifications.
- **Activities:** Preparing detailed documentation, creating reports, and presenting findings.
- **Status:** [Completed]

### 8. Future Directions and Scalability Assessment
- **Objective:** Explore potential future enhancements and assess scalability.
- **Activities:** Researching advanced techniques, assessing scalability, and drafting future work plans.
- **Status:** [Completed]

# Usage Example
## usage help
```
usage: vit.py [-h] [--gpu GPU] [--epochs EPOCHS] [--experts EXPERTS] [--batch BATCH] [--noswitch] [--cifar100] [--out OUT] [--dmodel DMODEL]

PyTorch Distributed deep learning

options:
  -h, --help         show this help message and exit
  --gpu GPU          no. of gpus
  --epochs EPOCHS    no. of epochs
  --experts EXPERTS  no. of experts
  --batch BATCH      batch size
  --noswitch         use original vit
  --cifar100         use cifar 100 dataset
  --out OUT          model output path
  --dmodel DMODEL    d_model embedding size
```

## examples

`python vit.py --batch 256 --experts 64 --epochs 500 --gpu 4 --dmodel 300 --out vit_model_rtx_4_dmodel_300_experts_64_batch_256_cifar10`<br>
Running switch-vit with 64 experts and 300 d_model embdedding size on 4 GPUs with 256 effective batch size over multiple GPU.


`python vit.py --batch 256 --experts 64 --epochs 500 --gpu 4 --dmodel 300 --noswitch --out vit_noswitch_model_rtx_4_dmodel_300_batch_256_cifar10`<br>
Running original vit with 64 experts and 300 d_model embdedding size on 4 GPUs with 256 effective batch size over multiple GPU.


# Results
model trained on cifar 10 for 350 epochs<br>
GPU: 4 RTX8000<br>
Batch size: 256<br>
Embedding Size: 300<br>
Patch Size: 4

t_epoch = Epoch to accuracy threshold 75%
| model | experts | size | highest acc (epoch) | t_epoch | Training Time (60,000 images) | Inference Time (60,000 images) |
|----|:----:|:----:|:-----:|:-----:|:-----:|:-----:|
| vit | - | 12,798,490 | 81.36 (345) | 31  | 11.8 | 4.6 |
| switch-vit | 32 | 238,732,426 | 79.04 (321) | 33 | 68.86 | 4.98 |
| switch-vit | 64 | 464,669,962 | 78.8 (328) | 49 | 130.52 | 5.18 |


# Citation
```
{
 author = {William Fedus, Barret Zoph, Noam Shazeer},
 title = {Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
 year = {2022},
 url = {https://arxiv.org/pdf/2101.03961.pdf},
}

@misc{labml,
 author = {Varuna Jayasiri, Nipun Wijerathne},
 title = {labml.ai: A library to organize machine learning experiments},
 year = {2020},
 url = {https://labml.ai/},
}
```
