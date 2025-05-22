from datasets import load_dataset
from sklearn.model_selection import train_test_split
from io import BytesIO
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


class EdgesToShoesDataset(Dataset):

    def __init__(self, hugf_dataset, transform=None):
        self.dataset = hugf_dataset
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, idx):
        image_a = Image.open(BytesIO(self.dataset[idx]['imageA']['bytes'])).convert("RGB")
        image_b = Image.open(BytesIO(self.dataset[idx]['imageB']['bytes'])).convert("RGB")
        
        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        return {'input': image_a, 'target': image_b}


if __name__ == "__main__":
    dataset = load_dataset("huggan/edges2shoes", split='train')
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=24, shuffle=True)
    train_data = dataset.select(train_indices)
    val_data = dataset.select(val_indices)

    train_dataset = EdgesToShoesDataset(train_data)
    val_dataset =   EdgesToShoesDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2)