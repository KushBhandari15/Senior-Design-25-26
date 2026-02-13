import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
from dice_coefficient import dice_coefficient, DiceBCELoss

class NoduleDataset(Dataset):
    def __init__(self, pos_dir, mask_dir):
        self.pos_files = sorted([os.path.join(pos_dir, f) for f in os.listdir(pos_dir)])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.pos_files)

    def __getitem__(self, idx):
        patch = np.load(self.pos_files[idx])[np.newaxis, ...]
        mask = np.load(self.mask_files[idx])[np.newaxis, ...]

        return torch.from_numpy(patch).float(), torch.from_numpy(mask).float()


class NoduleSegmentation(nn.Module):
    def __init__(self):
        super(NoduleSegmentation, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        # Bottleneck
        self.bottleneck = self.conv_block(32, 64)
        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = self.conv_block(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = self.conv_block(32 + 16, 16)
        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(out_c),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(out_c)
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        d2 = self.up2(b)
        if d2.shape != e2.shape: d2 = d2[:, :, :e2.shape[2], :e2.shape[3], :e2.shape[4]]
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape != e1.shape: d1 = d1[:, :, :e1.shape[2], :e1.shape[3], :e1.shape[4]]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


class Pipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NoduleSegmentation()
        self.model.to(self.device)

        dataset = NoduleDataset('data/pos', 'data/mask')
        total_count = len(dataset)
        train_count = int(total_count * 0.8); val_count = int(total_count * 0.1); test_count = int(total_count * 0.1)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_count, val_count, test_count],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def model_training(self, epochs=20):

        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device); labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}; Loss: {loss.item():.4f}")

    def validation_and_testing(self, event, loader):

        self.model.eval()
        total_loss = 0.0; total_dice = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device); labels = labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                total_dice += dice_coefficient(outputs, labels)

        avg_loss = total_loss / len(loader)
        avg_dice = total_dice / len(loader)
        print(f"{event} Loss {avg_loss:.4f}, Dice Score {avg_dice:.2f}%")
        return avg_dice

    def execute(self):
        print("--- Starting Training ---")
        self.model_training()
        print("\n--- Running Validation ---")
        self.validation_and_testing("Validation", self.val_loader)
        print("\n--- Final Testing ---")
        self.validation_and_testing("Testing", self.test_loader)
        print("\n--- Saving Model ---")
        torch.save(self.model.state_dict(), "nodule_segmentation_model.pth")

if __name__ == "__main__":
    Pipeline().execute()