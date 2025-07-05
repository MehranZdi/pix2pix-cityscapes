import torch
from dataset import CityscapesDataset
from torch.utils.data import DataLoader
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from trainer import train_model
import yaml
import time

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Data paths
    train_path = cfg["paths"]["train"]
    val_path = cfg["paths"]["val"]

    # Datasets
    train_dataset = CityscapesDataset(train_path, is_train=True)
    val_dataset = CityscapesDataset(val_path, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], num_workers=4)

    # Models
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg["optim"]["lr_g"], betas=cfg["optim"]["betas"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg["optim"]["lr_g"], betas=cfg["optim"]["betas"])

    # Start training
    start_time = time.time()
    train_model(generator, discriminator, train_loader, val_loader,
                optimizer_G, optimizer_D, device, cfg)
    end_time = time.time()
    print(f'Training completed in {(end_time - start_time):.2f} seconds')