import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from glob import glob
from transformers import SiglipVisionModel, SiglipImageProcessor
from tqdm import tqdm

# 1. Custom Dataset to handle the {split}/{lang}/{class} structure
class MemeDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Find all images in subfolders
        # Structure: root/split/lang/class/*.jpg
        for lang_dir in os.listdir(self.root_dir):
            lang_path = os.path.join(self.root_dir, lang_dir)     
            if not os.path.isdir(lang_path): continue
            
            for class_name in os.listdir(lang_path):
                class_path = os.path.join(lang_path, class_name)
                if not os.path.isdir(class_path): continue
                
                # Determine label: 1 for positive/postitive, 0 for negative
                label = 1 if "pos" in class_name.lower() else 0
                
                for img_ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    for img_path in glob(os.path.join(class_path, img_ext)):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
