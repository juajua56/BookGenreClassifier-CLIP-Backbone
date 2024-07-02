from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BookDataset(Dataset):
    def __init__(self, csv_df, transform):
        self.csv_df = csv_df
        self.img_paths = self.csv_df['Absolute_Path'].to_list()  # 이미지 경로만 저장
        self.text = self.csv_df['Title'].to_list()
        self.label = self.csv_df['label_index'].to_list()
        self.transform = transform
        self.img_cache = {}  # 이미지 캐싱을 위한 딕셔너리

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        if img_path not in self.img_cache:  # 캐시 확인
            img = Image.open(img_path).convert("RGB")
            self.img_cache[img_path] = img  # 캐시 저장

        img = self.img_cache[img_path]
        img = self.transform(img)
        return img, self.text[idx], self.label[idx]


def create_data_loader(df, preprocess, batch_size=20, shuffle=True, num_workers=8):
    dataset = BookDataset(df, transform=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

train_preprocess = transforms.Compose([
    transforms.Resize(size=(336, 336), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.ToTensor(),  # PILToTensor 제거 (ToTensor가 PIL 이미지를 처리)
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

val_preprocess = transforms.Compose([
    transforms.Resize(size=(336, 336), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.ToTensor(),  # PILToTensor 제거 (ToTensor가 PIL 이미지를 처리)
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])
