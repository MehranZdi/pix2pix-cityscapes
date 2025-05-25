import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import EdgesToShoesDataset
from sklearn.model_selection import train_test_split
from model import UNetGenerator, Discriminator
import torch.optim as optim
from tqdm import tqdm
import time


def train_one_epoch(
        generator,
        discriminator,
        train_loader,
        optimizer_G,
        optimizer_D,
        bce_loss,
        l1_loss,
        device,
        lambda_l1=100
):
    
    generator.train()
    discriminator.train()

    loop = tqdm(train_loader, desc="Training", leave=False)
    total_g_loss, total_d_loss = 0.0, 0.0

    for batch in loop:
        input_img = batch["input"].to(device)
        target_img = batch["target"].to(device)

        # Train Discriminator

        fake_img = generator(input_img)
        D_real = discriminator(torch.cat([input_img, target_img], dim=1))
        real_labels = torch.ones_like(D_real).to(device)
        loss_real = bce_loss(D_real, real_labels)
        
        # Fake pair (detach to avoid G updates)
        D_fake = discriminator(torch.cat([input_img, fake_img.detach()], dim=1))
        fake_labels = torch.zeros_like(D_fake).to(device)
        loss_fake = bce_loss(D_fake, fake_labels)

        d_loss = (loss_real + loss_fake) * 0.5
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator

        D_fake_for_G = discriminator(torch.cat([input_img, fake_img], dim=1))
        valid_labels = torch.ones_like(D_fake_for_G).to(device)

        g_adv_loss = bce_loss(D_fake_for_G, valid_labels)
        g_l1_loss = l1_loss(fake_img, target_img) * lambda_l1
        g_loss = g_adv_loss + g_l1_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        loop.set_postfix({
            "G_loss": f"{g_loss.item():.4f}",
            "D_loss": f"{d_loss.item():.4f}"
        })

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
    return avg_g_loss, avg_d_loss
    

def validate(generator, val_loader, l1_loss, device):
    val_l1_total = 0.0
    generator.eval()

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validation", leave=True)
        for batch in loop:
            input_img = batch["input"].to(device)
            target_img = batch["target"].to(device)

            fake_img = generator(input_img)
            l1 = l1_loss(fake_img, target_img)
            val_l1_total += l1.item()

            loop.set_postfix({"L1_loss": f"{l1.item():.4f}"})

        return val_l1_total / len(val_loader)


def train_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        optimizer_G,
        optimizer_D,
        device,
        num_epochs=20,
        lambda_l1=100,
        checkpoint_dir="checkpoints"
):

    writer = SummaryWriter(log_dir="runs/pix2pix_experiment")

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1} / {num_epochs}")
        avg_g_loss, avg_d_loss = train_one_epoch(
            generator, 
            discriminator,
            train_loader,
            optimizer_G,
            optimizer_D,
            bce_loss,
            l1_loss,
            device,
            lambda_l1
        )

        val_l1 = validate(generator, val_loader, l1_loss, device)

        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Val_L1", val_l1, epoch)

    
        print(f"Epoch {epoch+1} Summary: G_loss={avg_g_loss:.4f} | D_loss={avg_d_loss:.4f} | Val_L1={val_l1:.4f}")
        torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': avg_g_loss,
        'loss_D': avg_d_loss,
    }, 'checkpoint.pth')

        writer.close()

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The model is training on {device}...")
    dataset = load_dataset("huggan/edges2shoes", split="train")
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=24, shuffle=True)
    train_data = dataset.select(train_indices)
    val_data = dataset.select(val_indices)

    train_dataset = EdgesToShoesDataset(train_data)
    val_dataset = EdgesToShoesDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)
    
    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    start_time = time.time()
    train_model(generator, discriminator, train_loader,val_loader, optimizer_G, optimizer_D, device, num_epochs=15)
    end_time = time.time()
    print(f'Training took {(end_time - start_time):.4f}')