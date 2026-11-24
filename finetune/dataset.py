"""
dataset.py
Loads real/fake face images from an ImageFolder-style directory:

    data/
      train/real/*.jpg   train/fake/*.jpg
      val/real/*.jpg     val/fake/*.jpg
      test/real/*.jpg    test/fake/*.jpg

Uses the SAME resize/normalization as XceptionNetDeepfake in
backend/models/detector.py, so weights trained here are directly
compatible with the inference code.
"""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Must match backend/models/detector.py -> self.xception_transform
XCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Light augmentation for the training split only (helps a small fine-tune generalize)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


class DeepfakeFaceDataset(Dataset):
    """
    label 0 = real, 1 = fake  (must match detector.py's [real, fake] ordering)
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform or (TRAIN_TRANSFORM if split == 'train' else XCEPTION_TRANSFORM)

        self.samples = []
        for label_name, label_idx in [('real', 0), ('fake', 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}\n"
                    f"Directory must look like: data/{split}/real/ and data/{split}/fake/"
                )
            for fname in os.listdir(class_dir):
                if Path(fname).suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((str(class_dir / fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root_dir}")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[{split}] loaded {len(self.samples)} images "
              f"({n_real} real, {n_fake} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label
