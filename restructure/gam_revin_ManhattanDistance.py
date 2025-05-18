import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from utils.metrics import *
from torch.utils.data import TensorDataset, DataLoader, random_split
from revin.revin_ManhattanDistance import RevIN
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
# Fix random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Use any integer as seed
'''

Device = "cuda" if torch.cuda.is_available() else "cpu"
lambda_value = 1e-2
exp_itme = "jn_5"  # Just modify this to project_IMFcount, no other changes needed
type_names_list = ["jn", "nd", "ht"]

tensorset_dic = {
    "jn_3": 0,
    "jn_4": 1,
    "jn_5": 2,
    "jn_6": 3,
    "nd_3": 4,
    "nd_4": 5,
    "nd_5": 6,
    "ht_3": 7,
    "ht_4": 8,
    "ht_5": 9,
    "ht_6": 10,
}

type_key = exp_itme.split("_", 1)[0]
imf_nums = int(exp_itme.split("_", 1)[1])

input_dim = imf_nums + 1
hidden_dim = 64
out_dim = input_dim


class SignalReconstructor(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(SignalReconstructor, self).__init__()

        # Define normalization layer
        self.revinlayer = RevIN(num_features=input_dim)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, pred, imfs):
        # Concatenate input signals
        data = torch.cat([pred] + imfs, dim=1)
        # Normalize
        x_norm = self.revinlayer(data, mode='norm')
        # x = self.revinlayer(x_norm, mode='denorm')

        x = x_norm.unsqueeze(-1).unsqueeze(-1)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        x = x.squeeze(-1)

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        weights = torch.reshape(out, (hidden_dim, input_dim))
        weights = self.revinlayer(weights, mode='denorm')

        reconstructed_signal = weights[:, 0] * pred
        for i, imf in enumerate(imfs):
            reconstructed_signal += weights[:, i + 1] * imf
        # reconstructed_signal += weights[:, -1] * error  # Optional

        return reconstructed_signal, weights, data, x_norm


# Visualization function
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='viridis')
    plt.title(title)
    plt.show()


# Data parsing
def get_all_data_from_loader(dataloader):
    all_data = list(zip(*[batch for batch in dataloader]))
    all_data = [torch.cat(data, dim=0) for data in all_data]
    return all_data


def load_data(type_name: str, num_imf: int, file_path="./data/DYG_sgc_pred.xlsx") -> TensorDataset:
    try:
        df = pd.read_excel(file_path, sheet_name=f"{type_name}_{num_imf}")
        data_dic = {}
        data_dic["s_pred"] = torch.tensor(df[f"{type_name}_s_pred"].values).unsqueeze(1).to(torch.float32).to(Device)
        for i in range(num_imf):
            imf_column = f"sgc_{i}"
            if imf_column in df.columns:
                data_dic[imf_column] = torch.tensor(df[imf_column].values).unsqueeze(1).to(torch.float32).to(Device)
            else:
                raise ValueError(f"Column {imf_column} not found in {type_name} data.")
        # data_dic["error"] = torch.tensor(df["error"].values).unsqueeze(1).to(torch.float32).to(Device)  # Optional
        data_dic[f"{type_name}_true"] = torch.tensor(df[f"{type_name}_true"].values).unsqueeze(1).to(torch.float32).to(Device)
        dataset = TensorDataset(*data_dic.values())
        return dataset
    except Exception as e:
        print(f"Error for {type_name}: {e}")


def process_batch(batch_pram, num_imf: int, device):
    batch = [x.to(device) for x in batch_pram]
    s_pred = batch[0]
    imf_list = batch[1:num_imf + 1]
    # error = batch[num_imf + 1]  # Optional
    true = batch[num_imf + 1]
    return s_pred, imf_list, true


# Define minimal loss
min_loss = float('inf')
best_weights = None
model = SignalReconstructor(in_channels=input_dim, out_channels=out_dim).to(device=Device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()
data = load_data(type_key, imf_nums)
# Load all data at once for batch processing later

dataset = data
# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))  # Assume 80% for training
test_size = len(dataset) - train_size
# Sequential split of dataset
# train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
# test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training process
print(f"Training: Train item: {type_key}, IMF num: {imf_nums}")
for epoch in range(200):  # Train for 200 epochs
    model.train()
    for batch in train_loader:
        s_pred, imfs, true = process_batch(batch_pram=batch, num_imf=imf_nums, device=Device)
        optimizer.zero_grad()
        reconstructed_signal, weights, data, x_norm_m = model(s_pred, imfs)
        # print(weights)
        loss = mse_loss(reconstructed_signal, true)
        # Optionally add model complexity penalty
        # e.g., loss += lambda_value * torch.sum(weights**2)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()
            best_weights = weights.detach().clone()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Calculate metrics
weight = torch.mean(best_weights, dim=0, keepdim=True)
print(f"Minimum Loss: {min_loss}, Best Weights: {best_weights}, weight_mean: {weight}, data: {data}, x_norm_m: {x_norm_m}")

test_data_list = []

# Iterate over all test dataset samples
for i in range(len(test_dataset)):
    single_sample = test_dataset[i]
    single_sample_concatenated = torch.cat(single_sample, dim=0)  # Assume each sample is a tuple of tensors
    test_data_list.append(single_sample_concatenated)

# Stack all tensors in list into a large tensor
test_data_tensor = torch.stack(test_data_list)
test_feature = test_data_tensor[:, :-1]
test_true = test_data_tensor[:, -1].unsqueeze(1)
# Check shape of final tensor
print(test_data_tensor.shape)

# Manual weight adjustment (optional)
# weight = [0, 0.33, 0.33, 0.33, 0]
# weight = torch.tensor(weight).view(1, input_dim).to(device=Device)

re_signal = torch.matmul(test_feature, weight.t()).view(-1, 1)

test_true = test_true.cpu().detach().numpy()
re_signal = re_signal.cpu().detach().numpy()
mae_re, mse_re, rmse_re, mape_re, mspe_re = metric(pred=re_signal, true=test_true)
print(f"Result: Train item: {type_key}, IMF num: {imf_nums}")
print(f"mae_re: {mae_re:.4f}, mse_re: {mse_re:.4f}, rmse_re: {rmse_re:.4f}, mape_re: {mape_re:.4f}, mspe_re: {mspe_re}")
