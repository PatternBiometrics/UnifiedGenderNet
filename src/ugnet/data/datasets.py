from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class FaceHandCSV(Dataset):
    def __init__(self, csv_path, img_root, modality="hand", size=224, is_train=False):
        self.df = pd.read_csv(csv_path)
        self.root, self.modality = Path(img_root), modality
        self.y = (self.df["gender"].str.lower().map({"male":1,"female":0})).astype(int).values
        base = [T.Resize((size,size)), T.ToTensor()]
        aug = [T.RandomHorizontalFlip(), T.ColorJitter(0.1,0.1,0.05), T.RandomRotation(10)] if is_train else []
        self.tf = T.Compose(aug+base)

    def __len__(self): return len(self.df)
    def _path(self,i):
        col = "hand_image_name" if self.modality=="hand" else "face_image_name"
        sub = "hands" if self.modality=="hand" else "faces"
        return self.root/sub/self.df.iloc[i][col]
    def __getitem__(self,i):
        x = self.tf(Image.open(self._path(i)).convert("RGB"))
        y = torch.tensor(self.y[i]).long()
        m = torch.tensor([1,0] if self.modality=="hand" else [0,1]).float()
        return x,y,m
