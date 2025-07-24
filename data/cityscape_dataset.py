import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms


class CityscapesDataset(Dataset):
    def __init__(self, path, is_train=True, target_size=(256, 512)):
        self.img_dir = os.path.join(path, 'images')
        self.label_dir = os.path.join(path, 'masks')

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('_leftImg8bit.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('_gtFine_color.png')])

        # Pair images and labels by common prefix
        self.pairs = []
        img_prefixes = {f.split('_leftImg8bit')[0]: f for f in self.img_files}
        label_prefixes = {f.split('_gtFine_color')[0]: f for f in self.label_files}

        for prefix in img_prefixes:
            if prefix in label_prefixes:
                self.pairs.append((img_prefixes[prefix], label_prefixes[prefix]))

        assert len(self.pairs) > 0, "No matching image-label pairs found!"

        self.is_train = is_train
        self.target_size = target_size

        # transformations
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_filename, label_filename = self.pairs[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        img_tensor = self.transform(img)
        label_tensor = self.transform(label)
        return {"input": label_tensor, "target": img_tensor}  # label → photo


if __name__ == "__main__":
    train_path = "/home/mehran/Datasets/Cityscapes/train"
    val_path = "/home/mehran/Datasets/Cityscapes/val"

    train_dataset = CityscapesDataset(train_path, is_train=True, target_size=(256, 512))
    val_dataset = CityscapesDataset(val_path, is_train=False, target_size=(256, 512))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4)