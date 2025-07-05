import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from utils import load_checkpoint


def train_one_epoch(
        generator,
        discriminator,
        train_loader,
        optimizer_G,
        optimizer_D,
        bce_loss,
        l1_loss,
        device,
        lambda_l1=50,
        epoch=0
):
    generator.train()
    discriminator.train()

    loop = tqdm(train_loader, desc="Training", leave=False)
    total_g_loss, total_d_loss = 0.0, 0.0

    for batch_idx, batch in enumerate(loop):
        input_img = batch["input"].to(device)
        target_img = batch["target"].to(device)

        # Generate fake images
        fake_img = generator(input_img)

        ### Train Discriminator ###
        train_discriminator = True

        # Optional: Skip discriminator training for first few epochs to let generator warm up
        if epoch < 3:
            train_discriminator = False

        if train_discriminator:
            # Train on real images
            real_pair = torch.cat([input_img, target_img], dim=1)
            D_real = discriminator(real_pair)
            real_labels = torch.ones_like(D_real, device=device) * 0.9  # Label smoothing
            loss_real = bce_loss(D_real, real_labels)

            # Train on fake images
            fake_pair = torch.cat([input_img, fake_img.detach()], dim=1)
            D_fake = discriminator(fake_pair)
            fake_labels = torch.zeros_like(D_fake, device=device) + 0.1  # Label smoothing
            loss_fake = bce_loss(D_fake, fake_labels)

            # Total discriminator loss
            d_loss = (loss_real + loss_fake) * 0.5

            optimizer_D.zero_grad()
            d_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        ### Train Generator ###
        # Adversarial loss
        fake_pair_for_G = torch.cat([input_img, fake_img], dim=1)
        D_fake_for_G = discriminator(fake_pair_for_G)
        valid_labels = torch.ones_like(D_fake_for_G, device=device) * 0.9
        g_adv_loss = bce_loss(D_fake_for_G, valid_labels)

        # L1 loss
        g_l1_loss = l1_loss(fake_img, target_img) * lambda_l1

        # Total generator loss
        g_loss = g_adv_loss + g_l1_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        loop.set_postfix({
            "G_loss": f"{g_loss.item():.4f}",
            "D_loss": f"{d_loss.item():.4f}",
            "G_adv": f"{g_adv_loss.item():.4f}",
            "G_L1": f"{g_l1_loss.item():.4f}"
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


def save_sample_images(generator, val_loader, device, epoch, writer):
    generator.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        input_img = batch["input"].to(device)[:5]  # Take first 4 samples
        target_img = batch["target"].to(device)[:5]
        fake_img = generator(input_img)

        # Denormalize images for saving (from [-1,1] to [0,1])
        input_denorm = (input_img + 1) * 0.5
        target_denorm = (target_img + 1) * 0.5
        fake_denorm = (fake_img + 1) * 0.5

        # Create comparison grid
        comparison = torch.cat([input_denorm, fake_denorm, target_denorm], dim=0)
        grid = vutils.make_grid(comparison, nrow=5, normalize=False)

        # Save to tensorboard
        writer.add_image("Input_Generated_Target", grid, epoch)

        # Save to file
        vutils.save_image(grid, f"results/comparison_epoch_{epoch}.png")


def train_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        optimizer_G,
        optimizer_D,
        device,
        cfg
):
    writer = SummaryWriter(log_dir=cfg["training"]["log_dir"])

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    os.makedirs(cfg["paths"]["checkpoints"], exist_ok=True)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)

    start_epoch = 0
    best_val_l1 = float('inf')
    patience = 500  # Reduced patience
    counter = 0

    checkpoint_path = os.path.join(cfg["paths"]["checkpoints"], "checkpoint_epoch_331.pth")
    if os.path.exists(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        start_epoch, _, _ = load_checkpoint(
            generator, discriminator, optimizer_G, optimizer_D,
            checkpoint_path, device
        )
    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1} / {cfg['training']['epochs']}")

        avg_g_loss, avg_d_loss = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_G,
            optimizer_D,
            bce_loss,
            l1_loss,
            device,
            cfg["training"]["lambda_l1"],
            epoch
        )

        # Validation
        val_l1 = validate(generator, val_loader, l1_loss, device)

        # Log to tensorboard
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Val_L1", val_l1, epoch)
        # writer.add_scalar("Learning_Rate/Generator", optimizer_G.param_groups[0]['lr'], epoch)
        # writer.add_scalar("Learning_Rate/Discriminator", optimizer_D.param_groups[0]['lr'], epoch)

        # Save sample images
        if epoch % 5 == 0:
            save_sample_images(generator, val_loader, device, epoch, writer)

        print(f"Epoch {epoch + 1} Summary: G_loss={avg_g_loss:.4f} | D_loss={avg_d_loss:.4f} | Val_L1={val_l1:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_G': avg_g_loss,
                'loss_D': avg_d_loss,
            }, os.path.join(cfg["paths"]["checkpoints"], f"checkpoint_epoch_{epoch + 1}.pth"))


        # Early stopping and best model saving
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            counter = 0
            torch.save(generator.state_dict(),
                       os.path.join(cfg["paths"]["checkpoints"], f"best_generator_{epoch}.pth"))
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

        # Update learning rates every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            scheduler_G.step()
            scheduler_D.step()

    writer.close()