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

## Repository and Code Structure

### Repository Organization

#### Root Directory
- **Python Files:** All main Python files (.py) for the project are located here.
- **Run Scripts:** Scripts for running the project, facilitating easy execution.
- **Project Documentation:** Preliminary documentation available as PDF files.

#### `results` Directory
- Contains all output files generated from running the project scripts, including data files, logs, and output reports.

### Code Structure

#### Imports and Dependencies
- Standard libraries: `copy`, `time`, `random`.
- PyTorch modules: `torch`, `nn`, `torchvision`, `torch.distributed`, `torch.multiprocessing`.
- Helper libraries: `labml_helpers`, `labml_nn`, `argparse`, `typing`.

#### Model Components
- `Mlp`: Defines MLP used in Vision Transformer.
- `ParallelMLP`: Implements parallel MLPs with random routing and averaging.
- `PatchEmbeddings`: Generates patch embeddings from images.
- `LearnedPositionalEmbeddings`: Adds learned positional embeddings.
- `ClassificationHead`: MLP classification head for image classification.
- `Block`: Combines self-attention and feed-forward layers in a transformer block.
- `TransformerLayer`: Implements an encoder or decoder layer of a transformer.
- `VisionTransformer`: Main class combining all components for the Vision Transformer model.

#### Main Function (`main`)
- Initializes distributed processing.
- Sets up data loaders for CIFAR datasets.
- Constructs and configures the Vision Transformer model.
- Training and evaluation loops with functionality for model saving.

#### Utility Functions
- `get_total_params`: Calculates total parameters in the model.

#### Execution
- Parses command-line arguments.
- Starts distributed training using `mp.spawn`.

**Note:** The code is organized for modular design and flexibility, supporting distributed training and experimenting with innovative approaches like random routing in MLPs.



## Usage Example
### usage help
```
usage: vit.py [--gpu GPU] [--epochs EPOCHS] [--batch BATCH] [--cifar10] [--out OUT] [--dmodel DMODEL]

options:
  --gpu GPU          no. of gpus
  --epochs EPOCHS    no. of epochs
  --batch BATCH      batch size
  --cifar10          use cifar 100 dataset
  --out OUT          model output path
  --dmodel DMODEL    d_model embedding size
```
you can see example in run.sh


# Results
model trained on cifar 10 for 100 epochs<br>
GPU: 4 RTX8000<br>
Batch size: 256<br>
Embedding Size: 300<br>
Patch Size: 4


| model | experts | size | acc (epoch) | Training Time (60,000 images) |
|----|:----:|:----:|:-----:|:-----:|
| vit | - | 12,798,490 | 71.06 | 52.70 |
| vit-MoE | 4 | 12,812,938 | 71.1 | 71.80 |
| vit-MoE&Random Routing | 4 | 12,798,490 | 72.18 | 131.24 |


# Citation
```
@article{wang2022adamix,
  title={Adamix: Mixture-of-adapter for parameter-efficient tuning of large language models},
  author={Wang, Yaqing and Mukherjee, Subhabrata and Liu, Xiaodong and Gao, Jing and Awadallah, Ahmed Hassan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2205.12410},
  volume={1},
  number={2},
  pages={4},
  year={2022}
}
```
