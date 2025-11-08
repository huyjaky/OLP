from torch import nn
import pytorch_lightning as L
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv("./Attachments/data_train.csv")
# X = df.drop(columns=["shot_made_flag", "shot_id"]).values
X = torch.tensor(
    df[["loc_x", "loc_y", "minutes_remaining", "shot_distance"]].values,
    dtype=torch.float,
)
Y = df["shot_made_flag"].values

X[:, :2] = (
    2 * (X[:, :2] - (-224)) / (224 - (-224)) - 1
)  # Normalize loc_x and loc_y to [-1, 1]
X[:, -1] = 2 * (X[:, -1] - 0) / (79 - 0) - 1  # Normalize shot_distance to [-1, 1]


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)


class CustomCollator(Dataset):
    """
    Custom Dataset class to handle feature and label tensors.
    """

    def __init__(self, features, labels):
        self.features = features  # x_loc, y_loc, shot_distance, time_remaining
        self.labels = labels  # shot_made_flag

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float), torch.tensor(
            label, dtype=torch.float
        )


training_dataset = CustomCollator(X_train, Y_train)
validation_dataset = CustomCollator(X_val, Y_val)

training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


for sample in training_dataloader:
    features, labels = sample
    print(features.shape)  # torch.Size([32, 4])
    print(labels.shape)  # torch.Size([32])
    break
