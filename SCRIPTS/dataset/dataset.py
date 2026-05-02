import os
from PIL import Image
from torch.utils.data import Dataset

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.transform = transform
        self.input_dir = os.path.join(root_dir, mode, 'input')
        self.target_dir = os.path.join(root_dir, mode, 'target')
        self.files = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.files[idx])
        target_path = os.path.join(self.target_dir, self.files[idx])

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {"input": input_image, "target": target_image}
