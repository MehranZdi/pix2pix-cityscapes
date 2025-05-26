from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CityscapesDataset(Dataset):

    def __init__(self, path, transform=None):
        self.img_dir = os.path.join(path + '/img')
        self.label_dir = os.path.join(path + '/label')

        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.img_files)
    

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        image = self.transform(image)
        label = self.transform(label)

        return {'image': image, 'label': label}


if __name__ == "__main__":
    train_path = "/home/mehran/Documents/Pix2Pix/train"
    val_path = "/home/mehran/Documents/Pix2Pix/val"
    
    train_dataset = CityscapesDataset(train_path)
    val_dataset = CityscapesDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)