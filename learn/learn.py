# ============================================
# Step 1: ライブラリ読み込み + パス定義
# ============================================

import os
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ローカルのパスに変更（相対パスでOK）
train_img_dir = './data/cars_train/cars_train'
test_img_dir = './data/cars_test/cars_test'
anno_path = './data/car_devkit/devkit/cars_train_annos.mat'

# ============================================
# Step 2: ラベル読み込みとデータセット定義
# ============================================
data = scipy.io.loadmat(anno_path)['annotations'][0]

class StanfordCarsDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item[5][0]  # fname
        label = int(item[4][0]) - 1  # class (1-indexed -> 0-indexed)
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================
# Step 3: 前処理定義とデータローダ作成
# ============================================
transform = transforms.Compose([
    transforms.Resize((528, 528)),  # B6推奨サイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = StanfordCarsDataset(data, train_img_dir, transform)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # メモリに合わせ調整可

# ============================================
# Step 4: モデル準備（EfficientNetB6）
# ============================================
num_classes = len(set([int(item[4][0]) for item in data]))

model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ============================================
# Step 5: 学習
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {acc*100:.2f}%")

# ============================================
# Step 6: モデル保存（ローカル）
# ============================================
os.makedirs('./models', exist_ok=True)
torch.save(model.state_dict(), './models/efficientnetb6_car_model.pth')
print("モデル保存完了 → ./models/efficientnetb6_car_model.pth")
