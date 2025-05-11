<p align="center">
  <h1 align="center">PRNet: A Contrastive Ranking Model Based on 3D Convolution and Bi-LSTM for Channelrhodopsin Prediction</h1>
  <h2 align="center">🎉 Accepted at ICIC2025! 🎉</h2>
  <h3 align="center"><a href="URL">View Paper</a></h3>
</p>

![image](https://github.com/xi029/PRNet/blob/main/img/PRNet.png)



------

## 🎉 Introduction

**PRNet** is a deep learning framework designed to predict Channelrhodopsin (ChR) functional properties by fusing sequence and structural information. Leveraging a **Contrastive Ranking Strategy**, **3D Separable Convolution**, **Bi-LSTM**, and **SGA Attention**, PRNet excels on small-sample datasets, offering a powerful tool for optogenetics research.

> "Innovating with limited data for maximum impact."

------

## 📖 Table of Contents

- [🔧 Environment & Installation](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-environment--installation)
- [📂 Dataset Preparation](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-dataset-preparation)
- [🚀 Training & Evaluation](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-training--evaluation)
- [📑 Citation](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-citation)
- [🤝 Acknowledgements](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-acknowledgements)
- [📜 License](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-license)
- [⭐ Support PRNet](https://chatgpt.com/c/68205bdd-9590-800d-9736-8adb35d49be5#-support-prnet)

------

## 🔧 Environment & Installation

- **Hardware**: NVIDIA RTX 4090
- **Software**: Python 3.10, PyTorch 2.0.2

```bash
# 1. Clone
git clone https://github.com/yourusername/PRNet.git
cd PRNet

# 2. Virtual Environment
conda create -n PRNet python=3.10
conda activate PRNet

# 3. Dependencies
pip install -r requirements.txt
```

------

## 📂 Dataset Preparation

1. **Download** the ChR Functional Properties Dataset.

2. **Organize** in the following structure:

   ```
   dataset/
   ├── train/
   │   ├── sequences/
   │   ├── structures/
   │   └── labels/
   ├── val/
   │   ├── sequences/
   │   ├── structures/
   │   └── labels/
   └── test/
       ├── sequences/
       ├── structures/
       └── labels/
   ```

3. **Configure** paths in `config.yaml`.

------

## 🚀 Training & Evaluation

### Training

```bash
python train.py --config config.yaml
```

### Validation

```bash
python validate.py --config config.yaml
```

### Testing

```bash
python test.py --config config.yaml
```

------

## 📑 Citation

If PRNet benefits your work, please cite:

```bibtex
@inproceedings{yourpaper2025,
  title={{PRNet}: A Contrastive Ranking Model Based on 3D Convolution & Bi-LSTM for Channelrhodopsin Prediction},
  author={Your Name and Collaborators},
  booktitle={Proceedings of ICIC 2025},
  year={2025}
}
```

------

## 🤝 Acknowledgements

Thanks to:

- [Ultralytics](https://github.com/ultralytics)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [ParameterNet](https://github.com/parameter-net)
- [DEA-Net](https://github.com/dea-net)
- [RTDETR](https://github.com/rt-detr)

------

## 📜 License

This project is licensed under **GPL-3.0**. See [LICENSE](https://chatgpt.com/c/LICENSE).

------

## ⭐ Support PRNet

🚀 **欢迎 Star & Fork 支持本项目！**
 💖 **Feel free to ⭐️ and 🍴 this repo to contribute or build on PRNet!**
