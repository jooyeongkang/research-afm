import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


class ImageRegressionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()  # Default transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        target = torch.tensor(float(self.data_frame.iloc[idx, 1]), dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, target


class ModifiedResNet(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNet, self).__init__()
        self.base_model = base_model
        self.additional_layers = nn.Sequential(
            nn.Linear(base_model.fc.out_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):

        x = self.base_model(x)  # Pass through the original ResNet
        x = self.additional_layers(x)  # Pass through the custom layers

        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path="checkpoint.pt"):
        """
        Args:
            patience (int): How many epochs to wait for improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): Print early stopping messages.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Check if validation loss improved."""
        score = -val_loss  # We minimize loss, so higher negative is better

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the current best model."""
        if self.verbose:
            print(f"Validation loss decreased. Saving model...")
        torch.save(model.state_dict(), self.path)