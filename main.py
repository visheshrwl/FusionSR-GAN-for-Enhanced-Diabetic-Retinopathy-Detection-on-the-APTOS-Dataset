import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

print("Loading dataset paths and labels")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/sample_submission.csv")

print("Setting up image transformations")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class APTOSDataset(data.Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        print(f"Initializing APTOSDataset with {len(dataframe)} images in {img_dir}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0] + ".png")
        print(f"Loading image {img_name}")
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 1] if 'diagnosis' in self.dataframe.columns else -1
        
        if self.transform:
            image = self.transform(image)
        return image, label

print("Creating train dataset and loader")
train_dataset = APTOSDataset(train_df, 'data/train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class CNNEnhancementBlock(nn.Module):
    def __init__(self):
        super(CNNEnhancementBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        print("Initialized CNNEnhancementBlock")

    def forward(self, x):
        return self.conv(x)

class AutoencoderGANFusion(nn.Module):
    def __init__(self):
        super(AutoencoderGANFusion, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        print("Initialized AutoencoderGANFusion")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DualGANRefinement(nn.Module):
    def __init__(self):
        super(DualGANRefinement, self).__init__()
        self.detail_gan = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.perceptual_gan = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        print("Initialized DualGANRefinement")

    def forward(self, x):
        detail_refined = self.detail_gan(x)
        perceptual_refined = self.perceptual_gan(detail_refined)
        return perceptual_refined

class FusionSRGANClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(FusionSRGANClassifier, self).__init__()
        self.cnn_block = CNNEnhancementBlock()
        self.autoencoder_block = AutoencoderGANFusion()
        self.gan_refinement = DualGANRefinement()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        print("Initialized FusionSRGANClassifier")

    def forward(self, x):
        x = self.cnn_block(x)
        x = self.autoencoder_block(x)
        x = self.gan_refinement(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = FusionSRGANClassifier().to(device)

print("Setting up loss function and optimizer")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting training loop")
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loader_progress = tqdm(train_loader, desc=f"Training Batch", leave=False)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        train_loader_progress.set_postfix(loss=loss.item())
    
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

print("Saving model as FusionSRGAN_Classifier.pth")
torch.save(model.state_dict(), "FusionSRGAN_Classifier.pth")

print("Loading model for inference")
model = FusionSRGANClassifier(num_classes=5)
model.load_state_dict(torch.load("FusionSRGAN_Classifier.pth"))
model.eval()

print("Preparing test dataset and loader")
test_dataset = APTOSDataset(test_df, 'data/test_images', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Starting inference on test set")
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())

print("Saving predictions to submission.csv")
submission_df = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': predictions})
submission_df.to_csv("data/submission.csv", index=False)
print("Submission file created as submission.csv")
