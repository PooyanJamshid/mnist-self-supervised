# 🧠 Self-Supervised Learning on MNIST

This repository showcases a self-supervised learning pipeline applied to the MNIST dataset using multiple algorithms, all implemented from scratch with PyTorch. The goal is to learn meaningful representations **without using labels**, and later evaluate them via classification.

---

## 📌 Overview

Implemented self-supervised methods:

- ✅ **Autoencoder** (Encoder-Decoder) — learns by reconstructing input images.
- 🔄 **Rotation Prediction** — trains a model to predict the rotation angle of images (0°, 90°, 180°, 270°).
- 🔗 **SimCLR (Contrastive Learning)** — learns by contrasting positive and negative pairs using `NTXentLoss` from the [Lightly](https://github.com/lightly-ai/lightly) library.

Each method is evaluated by:
- Visualizing **loss curves**
- Measuring **per-class accuracy**
- Applying **dimensionality reduction (PCA / t-SNE)** to visualize latent space
- Showing **class distribution balance**

---


## 📦 Requirements

Install all dependencies using pip:

```bash
pip install torch torchvision matplotlib scikit-learn lightly tqdm
```

---

## 📂 Dataset

- **MNIST** (28x28 grayscale handwritten digits)
- Loaded via `torchvision.datasets.MNIST`
- Augmentations used:
  - Random rotations (for rotation task)
  - Cropping, flipping, jittering (for SimCLR)
- The dataset is balanced, and class distributions were visualized to confirm it.

---

## 🧪 Method Details

### 1. Autoencoder
- Architecture: Simple CNN encoder + symmetric decoder
- Trained using MSE loss between input and reconstructed output
- No labels used

### 2. Rotation Prediction
- Input images are rotated randomly
- Model is trained to predict one of 4 rotation classes
- Supervision is self-generated from the rotation itself

### 3. SimCLR
- Uses two random augmentations per image to create a positive pair
- Contrastive loss: NT-Xent (`lightly.loss.NTXentLoss`)
- Negative pairs sampled within batch

---

## 📊 Evaluation

- 📉 **Loss Curves**: Each model’s training loss was tracked and plotted
- 🎯 **Accuracy Per Class**: Evaluated using a linear classifier on top of the frozen encoder
- 🌈 **Latent Space Visualization**: PCA and t-SNE used to inspect learned representations
- 📊 **Class Distribution**: Verified uniformity across classes before training

---

## 📈 Results Summary

| Method            | Stable Loss | Accuracy | Notes                          |
|-------------------|-------------|----------|--------------------------------|
| Autoencoder       | ✅ Yes       | High   | Good for simple representations |
| Rotation          | ⚠️ Noisy     | Medium   | Struggles with similar digits (e.g. 3 vs 9) |
| SimCLR            | ✅ Stable    | Medium     | Best generalization & low overfitting |

---

## 🚧 Future Work

- Use CIFAR-10 or SVHN for more complex experiments
- Apply deeper architectures like ResNet
- Add semi-supervised finetuning

---

## 👤 Author

**Pooyan Jamshidi**  
A self-supervised learning enthusiast exploring representation learning with simple datasets and custom implementations.

---

## 📝 License

This project is open-source and available under the MIT License.
