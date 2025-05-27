import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import CityscapesDataset
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
        lambda_l1=50,  # lowered
        epoch=10
):
    generator.train()
    discriminator.train()

    loop = tqdm(train_loader, desc="Training", leave=False)
    total_g_loss, total_d_loss = 0.0, 0.0

    for batch in loop:
        input_img = batch["input"].to(device)
        target_img = batch["target"].to(device)

        fake_img = generator(input_img)

        ### Train Discriminator
        if epoch > 30:  # Freeze D for first 5 epochs
            D_real = discriminator(torch.cat([input_img, target_img], dim=1))
            real_labels = torch.full_like(D_real, 0.9).to(device)  # label smoothing
            loss_real = bce_loss(D_real, real_labels)

            D_fake = discriminator(torch.cat([input_img, fake_img.detach()], dim=1))
            fake_labels = torch.zeros_like(D_fake).to(device)
            loss_fake = bce_loss(D_fake, fake_labels)

            d_loss = (loss_real + loss_fake) * 0.5
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
        else:
            d_loss = torch.tensor(0.0)  # D not trained yet

        ### Train Generator
        D_fake_for_G = discriminator(torch.cat([input_img, fake_img], dim=1))
        valid_labels = torch.ones_like(D_fake_for_G).to(device)
        g_adv_loss = bce_loss(D_fake_for_G, valid_labels)
        g_l1_loss = l1_loss(fake_img, target_img) * lambda_l1
        g_loss = g_adv_loss + g_l1_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Confirm G is updating
        # for name, param in generator.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad mean: {param.grad.abs().mean().item()}")
        #         break

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
        num_epochs=200,
        lambda_l1=50,
        checkpoint_dir="checkpoints"
):
    writer = SummaryWriter(log_dir="runs/pix2pix_experiment")

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    os.makedirs(checkpoint_dir, exist_ok=True)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 if epoch < 100 else 1 - (
                epoch - 100) / 100)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 1.0 if epoch < 100 else 1 - (
                epoch - 100) / 100)

    best_val_l1 = float('inf')
    patience = 80
    counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} / {num_epochs}")
        avg_g_loss, avg_d_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_G,
            optimizer_D,
            bce_loss,
            l1_loss,
            device,
            lambda_l1,
            epoch  # ← pass current epoch
        )

        val_l1 = validate(generator, val_loader, l1_loss, device)

        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Val_L1", val_l1, epoch)

        if epoch % 5 == 0:
            with torch.no_grad():
                batch = next(iter(val_loader))
                input_img = batch["input"].to(device)[:4]
                fake_img = generator(input_img)
                grid = vutils.make_grid(fake_img, normalize=True)
                writer.add_image("Generated Images", grid, epoch)

        # if epoch % 10 == 0:
        #     vutils.save_image(fake_img * 0.5 + 0.5, f"results/fake_epoch_{epoch}.png")

        print(f"Epoch {epoch + 1} Summary: G_loss={avg_g_loss:.4f} | D_loss={avg_d_loss:.4f} | Val_L1={val_l1:.4f}")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'loss_G': avg_g_loss,
            'loss_D': avg_d_loss,
        }, f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth')

        scheduler_G.step()
        scheduler_D.step()

        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

    writer.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The model is training on {device}...")

    train_path = "/home/mehran/Datasets/Pix2Pix/Cityscapes/train"
    val_path = "/home/mehran/Datasets/Pix2Pix/Cityscapes/val"

    train_dataset = CityscapesDataset(train_path, is_train=True)
    val_dataset = CityscapesDataset(val_path, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))  # Reduced D learning rate
    start_time = time.time()
    train_model(generator, discriminator, train_loader, val_loader, optimizer_G, optimizer_D, device, num_epochs=200)
    end_time = time.time()
    print(f'Training took {(end_time - start_time):.4f}')