# Pix2Pix Cityscapes (Semantic → Photo Translation)

A PyTorch implementation of the **Pix2Pix** model for paired image-to-image translation on the **Cityscapes** dataset. This project transforms semantic label maps into realistic RGB street scenes using a U-Net Generator and a PatchGAN Discriminator.

---

## 🔧 Features

* ✅ Modular code (models, training, data, configs)
* ✅ Self-attention U-Net Generator
* ✅ PatchGAN Discriminator
* ✅ TensorBoard visualizations
* ✅ Reproducible experiments

---

## 📁 Project Structure

```
pix2pix_project/
├── configs/              # YAML configuration files
├── data/                 # Dataset class (Cityscapes)
├── models/               # Generator, Discriminator, Blocks
├── training/             # Trainer, utils
├── scripts/              # Training entry point
├── checkpoints/          # The best checkpoint
├── results/              # Generated sample images
├── runs/                 # TensorBoard logs
├── requirements.txt
└── README.md
```

---

## 🧪 Example Results

| Input (Label)                 | Generated                    | Target (Photo)               |
| ----------------------------- | ---------------------------- | ---------------------------- |
| ![](results/sample_input.png) | ![](results/sample_fake.png) | ![](results/sample_real.png) |

More samples are saved every 5 epochs to the `results/` folder.

---

## 🚀 Getting Started

### 1. Clone the repo

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Make sure your Cityscapes dataset is structured as:

```
Cityscapes/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
```

Each image should follow the naming:

* Input: `*_gtFine_color.png`
* Target: `*_leftImg8bit.png`

Update the paths in `configs/default.yaml` accordingly.

### 4. Start Training

```bash
python scripts/train.py
```

Training progress and generated samples will be saved to:

* TensorBoard logs: `runs/`
* Checkpoints: `checkpoints/`
* Samples: `results/`

---

## ⚙️ Configuration (YAML)

You can adjust training and model parameters in:

```
configs/default.yaml
```

Examples:

```yaml
training:
  epochs: 400
  batch_size: 8
  lambda_l1: 50

optim:
  lr_g: 0.0003
  lr_d: 0.0002
  betas: [0.5, 0.999]
```

---

## 📊 TensorBoard

Launch TensorBoard:

```bash
tensorboard --logdir runs/
```
## Docker Image

You can also use the docker image available on Dockerhub

```
docker pull mehranzdi/pix2pix-cityscapes
```

---

## 🧑‍💻 Author

* GitHub: [MehranZdi](https://github.com/MehranZdi)
---

## 📄 License

MIT License.
