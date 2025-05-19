# SimCLRv2-Pytorch

This repository provides a **PyTorch implementation** of the [SimCLR](https://arxiv.org/abs/2002.05709) framework:  
**"A Simple Framework for Contrastive Learning of Visual Representations"** by Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.

The original implementation was developed in **TensorFlow 2** and is available here:  
🔗 [google-research/simclr](https://github.com/google-research/simclr)

---

## About This Repository

The original SimCLR repository contains the TensorFlow 2 implementation under the `tf2/` directory.  
In this project, we provide a PyTorch-based reimplementation located in the `pytorch/` directory.

- ✅ PyTorch-based code for SimCLR
- 🧪 Compatible with standard datasets (e.g., CIFAR-10, STL-10, ImageNet)
- 🔁 Follows the structure and logic of the original `tf2/` code

---

## ⚠️ Note on Original Code Modifications

To facilitate comparison between the original TensorFlow 2 implementation and this PyTorch version,  
**minor changes may have been made to the files in the `tf2/` directory** (e.g., logging, output formatting, shared configuration).  
These edits are **minimal and only for compatibility/testing purposes**. The core logic remains unchanged.

---

## Directory Structure
```bash
simclr-pytorch/
├── model.py
├── run.py
├── pytorch/             # PyTorch implementation of SimCLR
│   ├── model.py
│   ├── run.py
│   └── ...
├── tf2/                 # Slightly modified original TensorFlow 2 implementation
├── README.md
├── LICENSE              # MIT License from the original repository
└── requirements.txt
```
## Acknowledgements

Thanks to the authors of SimCLR and the maintainers of the original TensorFlow implementation at [google-research/simclr](https://github.com/google-research/simclr).


