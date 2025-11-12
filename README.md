# Benchmarks for 3DFDReal Evaluation

> Real-World 3D Fashion Dataset Benchmark Evaluation

---

## 📚 Overview
- This repository provides **reproducible semantic segmentation code** for the **3DFDReal** dataset.
- The training/evaluation pipeline is built on **Pointcept**, including preprocessing scripts, configuration files, and example checkpoints.
- Other tasks (reconstruction, transformation, etc.) and additional models will be released progressively.

---

## 🔍 What’s in this release
- ✅ Semantic Segmentation (Pointcept-based)
- ✅ Data preprocessing and example folder structure
- ✅ Training/evaluation scripts and configuration templates
- ❌ SDVformer / text-shape alignment modules (to be released later)
- ❌ Real-time demo code (to be released later)

---

## 📦 Dataset
### 3DFDReal (recommended)
- **Real-World 3D Fashion Dataset (3DFDReal)**
- Contains multiple garment combinations, real-world occlusion, and multimodal alignment across point clouds/videos/metadata.
- Download: Hugging Face — 3DFDReal  
  👉 https://huggingface.co/datasets/kusses/3DFDReal
- **Dataset access and usage terms are provided directly on the Hugging Face dataset page. Please check the link above.**

---

## 🗂️ Repository Structure
```
3DFDReal-Pointcept/
├── configs/
│   ├── semseg-pt-v3m1-0-base-3dfdreal.py      # Training/eval config
├── pointcept				   
│   ├── datasets/                              
│   │   ├── 3dfdreal.py                        # Data preprocessing
│   ├── engines/
│   ├── models/
│   └── utils/
├─ tools/
│   ├── train.py                               # Training script (Pointcept wrapper)
│   ├── test.py                                # Evaluation script (mIoU, OA, mAcc)
│   └── data_split.py                          # Data split
└── README.md                                  # Pointcept readme


```

---

## ⚙️ Setup
### 1) Clone
```bash
git clone https://github.com/kusses/3DFDReal-SemSeg.git
cd 3DFDReal-SemSeg
```

### 2) Conda & PyTorch
```bash
conda create -n 3dfdreal python=3.10 -y
conda activate 3dfdreal
# Example: CUDA 12.x (adjust to your system)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 3) Pointcept & other dependencies
```bash
cd pointcept
pip install -r requirements.txt   # pointcept, open3d, numpy, pyyaml, etc.
pip install -e ./pointcept        # if included, or add as submodule
```

---

## 🧰 Data Preparation
Example structure:
```
/data/3dfdreal/
├── trains/                           # coord.npy, color.npy, normal.npy. segment.npy(class)
├── test/                             # coord.npy, color.npy, normal.npy. segment.npy(class)
├── val/                              # coord.npy, color.npy, normal.npy. segment.npy(class)
└── splits/                           # train.txt / val.txt / test.txt
```


## 🔥 Training
Training:
```bash
python tools/train.py \
  --config configs/3dfdreal.pointcept.semseg.yaml \
  --save-dir outputs/ptv3_semseg
```
Resume:
```bash
python tools/train.py --config configs/3dfdreal.pointcept.semseg.yaml \
  --resume outputs/ptv3_semseg/ckpt_latest.pth
```

---

## 📈 Evaluation
- Main metrics: **mIoU**, **OA (Overall Accuracy)**, **mAcc**
- Optional: per-class **AP** for imbalance analysis

Run train:
```bash
python Poinctcept/tools/train
--config-file
[your path]/Pointcept/configs/3dfdreal/semseg-pt-v3m1-0-base-3dfdreal.py 
--options
save_path=[your path]/Pointcept/exp/3dfdreal/semseg-pt-v1m1_3dfdreal
--num-gpus
1
```

Run evaluation:
```bash
python Poinctcept/tools/test
--config-file
[your path]/Pointcept/configs/volmeda/semseg-pt-v1m1_volmedata.py
--options
save_path=[your path]/Pointcept/exp/3dfdreal/semseg-pt-v1m1_3dfdreal
weight=[your path]/Pointcept/exp/3dfdreal/semseg-pt-v1m1_3dfdreal/model/model_best.pth

Visualization (optional):
```
---

## 🧾 Citation
Please cite our work when using this code or dataset:
```bibtex
@article{Lim2026DFDReal,
  title   = {3DFDReal: Real-World 3D Fashion Dataset for Virtual Try-On Applications},
  author  = {Lim, Jiyoun and et al.},
  journal = {TB},
  year    = {2025},
  note    = {under review}
}
```

---

## 🔗 Related Works
- [Pointcept](https://github.com/Pointcept/PointTransformerV3.git)
- [SVDformer](https://github.com/czvvd/SVDFormer_PointSea)
---

## 🤝 Contributing
- Issues/PRs are welcome. Please report reproduction issues, label mismatches, or metric calculation bugs.
- Code style: `black`, `isort` recommended.

---

## 📄 License
**3DFDReal dataset/annotations**: CC BY-NC 4.0 (non-commercial use only)
> Note: The dataset usage terms are specified on the official Hugging Face dataset page. Always review and comply with them.

%---

%## 👥 Authors
%- **Jiyoun Lim (@kusses), Jeong-Woo Son, Alex Lee, Sun-Joong Kim, NamKyung Lee, Wonjoo Park**

## 🙏 Acknowledgments
- Thanks to the Pointcept team for their excellent framework.
- Thanks to the open-source 3D fashion/point cloud community.

---

## 🗺️ Roadmap
- [ ] Additional backbones (PTv3m0/m1) and hyperparameter recipes
- [ ] Real-time demo / WebGL visualization examples
- [ ] SDVformer and text-shape alignment modules
- [ ] Pre-trained checkpoints

