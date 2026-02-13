import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class NoduleDataset(Dataset):
    def __init__(self, pos_dir, neg_dir):
        self.pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir)]
        self.neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)]
        self.all_files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        patch = np.load(file_path)
        patch = patch[np.newaxis, ...]

        patch_tensor = torch.from_numpy(patch).float()
        label_tensor = torch.tensor(self.labels[idx]).long()

        return patch_tensor, label_tensor

class NoduleDetection(nn.Module):
    def __init__(self):
        super(NoduleDetection, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Pipeline:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NoduleDetection().to(self.device)

        # DataLoader
        dataset = NoduleDataset('data/pos', 'data/neg')
        total_count = len(dataset)
        train_count = int(total_count * 0.8); val_count = int(total_count * 0.1); test_count = total_count - train_count - val_count
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_count, val_count, test_count], generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 32 batches and shuffling
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        self.criterion = nn.CrossEntropyLoss() # Use Cross-Entropy-Loss as the loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001) # Use adam optimizer

    def model_training(self):

        self.model.train()
        for epoch in range(20):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad() # Clear memory
                outputs = self.model(images) # Forward pass
                loss = self.criterion(outputs, labels) # Calculate error
                loss.backward() # Backward loss
                self.optimizer.step() # Update gradients
            print(f"Epoch {epoch+1}/15, Loss: {loss.item()}")


    def validation_and_testing(self, event, loader):

        self.model.eval()
        val_loss = 0; correct = 0; total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = val_loss / len(loader)
            accuracy = (correct / total) * 100
            print(f"{event} Loss {avg_loss:.4f}, Accuracy {accuracy:.2f}%")
            return avg_loss

    def execute(self):
        print("--- Starting Training ---")
        self.model_training()
        print("\n--- Running Validation ---")
        self.validation_and_testing("Validation", self.val_loader)
        print("\n--- Final Testing ---")
        self.validation_and_testing("Testing", self.test_loader)
        print("\n--- Saving Model ---")
        torch.save(self.model.state_dict(), "nodule_detection_model.pth")

if __name__ == "__main__":
    Pipeline().execute()