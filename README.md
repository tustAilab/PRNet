# PRNet: A Contrastive Ranking Model Based on 3D Convolution and Bi-LSTM for Channelrhodopsin Prediction

![image](https://github.com/xi029/PRNet/blob/main/img/PRNet.png)

## Overview

Channelrhodopsins (ChRs) are pivotal tools in optogenetics, enabling precise modulation of neuronal circuits. Traditional experimental methods for screening high-performance ChR variants are both costly and time-consuming, compounded by the scarcity of experimental data. To address these challenges, we introduce PRNet, a deep learning framework that integrates contrastive ranking with 3D convolutional networks and Bi-LSTM architectures. PRNet effectively predicts ChR functional properties using limited sample sizes, offering a robust solution for small-sample learning and multimodal feature fusion.

## Features

- **Contrastive Ranking Strategy**: Generates ample sample pairs from limited data to enhance model training.
- **Multimodal Feature Fusion**: Jointly encodes ChR sequence and structural information for comprehensive representation.
- **Efficient High-Dimensional Feature Extraction**: Employs depthwise separable 3D convolutions for structural features and Bi-LSTM for sequence dependencies.
- **Attention Mechanism**: Integrates SGA attention to capture intricate relationships between amino acids, boosting prediction accuracy.

## How to Use

### Environment Setup

- **Hardware**: NVIDIA RTX 4090
- **Software**:
  - Python 3.10
  - PyTorch 2.0.2

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/PRNet.git
   cd PRNet

   ```

1. **Create and activate a Python virtual environment**:

   ```
   conda create -n PRNet python=3.10
   conda activate PRNet
   ```

1. **Install the required dependencies**:

   ```
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. **Obtain the ChR dataset**:

   - Download the dataset from ChR Functional Properties Dataset.

2. **Organize the dataset**:

   ```
   pgsql复制编辑├── dataset/
       ├── train/
           ├── sequences/
           ├── structures/
           ├── labels/
       ├── val/
           ├── sequences/
           ├── structures/
           ├── labels/
       ├── test/
           ├── sequences/
           ├── structures/
           ├── labels/
   ```

3. **Update the dataset path**:

   - Modify the configuration file (`config.yaml`) to reflect the correct dataset paths.

### Training

To train PRNet from scratch:

```
python train.py --config config.yaml
```

### Validation

To validate the trained model:

```
python validate.py --config config.yaml
```

### Testing

To test the model:

```
python test.py --config config.yaml
```

## Acknowledgements

We extend our gratitude to the developers of the following projects for their valuable contributions:

- [Ultralytics](https://github.com/ultralytics)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [ParameterNet](https://github.com/parameter-net)
- [DEA-Net](https://github.com/dea-net)
- [RTDETR](https://github.com/rt-detr)

## Citation

If you find PRNet useful in your research, please cite our paper:

```
@article{yourpaper2025,
  title={PRNet: A Contrastive Ranking Model Based on 3D Convolution and Bi-LSTM for Channelrhodopsin Prediction},
  author={Your Name and Collaborators},
  journal={Journal/Conference Name},
  year={2025}
}

```

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
