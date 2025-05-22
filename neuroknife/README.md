# 🧠 neuroknife

**neuroknife** is a modular, extensible Python toolkit designed to simplify common side-tasks in Deep Learning projects — such as data loading, augmentations, model definitions, loss functions, exports, and utility logic.

> Like a Swiss Army Knife — but for neural networks.

---

## ✨ Why "neuroknife"?

The name **neuroknife** blends two ideas:
- **"Neuro"**: Refers to neural networks and deep learning
- **"Knife"**: Inspired by the Swiss Army Knife — a multi-tool concept that equips you with everything you need to handle deep learning side-work efficiently

---

## 🎯 What is this for?

This is **not a training framework**. Instead, `neuroknife` is a utility toolbox you can use in your own training pipelines, whether you're using PyTorch or TensorFlow.

It’s ideal for:
- Loading and preparing datasets
- Augmenting inputs with a consistent API
- Defining and registering models for PyTorch & TensorFlow
- Managing loss functions
- Exporting trained models (e.g. ONNX, TorchScript)
- Centralizing reusable components (schedulers, metrics, config loaders)

---

## 📦 What's Inside?

| Module         | Description |
|----------------|-------------|
| `data/`        | Dataset loaders, augmentations, preprocessing |
| `models/`      | Model architectures for PyTorch and TensorFlow |
| `losses/`      | Loss functions, both custom and built-in wrappers |
| `training/`    | Training utilities (not loops, but tools) |
| `exporting/`   | Functions for exporting trained models |
| `utils/`       | Config handling, logging, metrics |
| `registry.py`  | Plugin system for models, losses, etc. |

---

## 🗂️ Folder Structur

```bash
neuroknife/
├── data/ # Data loading, augmentations
├── models/ # Model definitions + registry
│ ├── pytorch/ # PyTorch-specific models
│ └── tensorflow/ # TensorFlow-specific models
├── losses/ # Loss functions + registry
├── training/ # Utility logic for schedulers, etc.
├── exporting/ # Export functions for models
├── utils/ # Logging, metrics, configs
├── init.py
├── pyproject.toml # Project and dependency configuration
├── README.md
├── LICENSE
└── .gitignore
```
---

## 🔧 Installation (Editable)

```bash
uv pip install -e .
```
