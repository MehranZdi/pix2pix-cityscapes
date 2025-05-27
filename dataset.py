from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random

class CityscapesDataset(Dataset):
    def __init__(self, path, is_train=True):
        self.img_dir = os.path.join(path, 'img')
        self.label_dir = os.path.join(path, 'label')

        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        assert len(self.img_files) == len(self.label_files), "Image and label count mismatch!"

        self.is_train = is_train
        self.resize = transforms.Resize((286, 286))
        self.crop = transforms.RandomCrop((256, 256))
        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.is_train:
            # Apply same random operations to both input & label
            img, label = self.resize(img), self.resize(label)

            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(256, 256))
            img = transforms.functional.crop(img, i, j, h, w)
            label = transforms.functional.crop(label, i, j, h, w)

            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                label = transforms.functional.hflip(label)

        else:
            img = transforms.Resize((256, 256))(img)
            label = transforms.Resize((256, 256))(label)

        img = self.to_tensor(img)
        label = self.to_tensor(label)

        img = self.normalize(img)
        label = self.normalize(label)

        return {"input": label, "target": img}  # ← Pix2Pix: label → photo


if __name__ == "__main__":
    train_path = "/home/mehran/Datasets/Pix2Pix/Cityscapes/train"
    val_path = "/home/mehran/Datasets/Pix2Pix/Cityscapes/train"

    train_dataset = CityscapesDataset(train_path)
    val_dataset = CityscapesDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)