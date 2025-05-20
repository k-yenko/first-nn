import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re

# load data
data = pd.read_csv('data/processed/raf_features.csv')

def parse_hgvs_multi(hgvs_str):
    """
    Parse HGVS string to extract up to two (wt, position, mut) tuples.
    Returns: [(wt1, pos1, mut1), (wt2, pos2, mut2)]
    If only one mutation, second tuple is (None, None, None).
    """
    aa_map = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
        'Ter': 'X'  # for stop codon
    }
    # Match single or multiple mutations
    pattern = r'([A-Z][a-z][a-z])([0-9]+)([A-Z][a-z][a-z]|Ter)'
    matches = re.findall(pattern, hgvs_str)
    muts = []
    for m in matches[:2]:  # Only take up to 2
        wt, pos, mut = m
        muts.append((aa_map.get(wt, 'X'), int(pos), aa_map.get(mut, 'X')))
    # Pad with (None, None, None) if only one mutation
    while len(muts) < 2:
        muts.append((None, None, None))
    return muts

def encode_fixed_size(df, protein_length=188):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    features = []
    mutation_counts = []
    for hgvs in df['hgvs_pro']:
        muts = parse_hgvs_multi(hgvs)
        row_feats = []
        count = 0
        for wt, pos, mut in muts:
            # One-hot for WT
            wt_vec = [0]*20
            if wt in aa_to_index:
                wt_vec[aa_to_index[wt]] = 1
            # One-hot for MUT
            mut_vec = [0]*20
            if mut in aa_to_index:
                mut_vec[aa_to_index[mut]] = 1
            # Normalized position
            pos_norm = (pos / protein_length) if pos is not None else 0
            # Add to row
            row_feats.extend(wt_vec + mut_vec + [pos_norm])
            if wt is not None:
                count += 1
        features.append(row_feats)
        mutation_counts.append(count)
    # Add mutation count as a feature
    features = np.array(features)
    mutation_counts = np.array(mutation_counts).reshape(-1, 1)
    features = np.hstack([features, mutation_counts])
    return features

# Build new feature matrix
X = encode_fixed_size(data)
y = data['score'].values

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Feature stats:", np.min(X), np.max(X), np.mean(X), np.std(X))
print("Target stats:", np.min(y), np.max(y), np.mean(y), np.std(y))

# create dataset and dataloader
class RAFDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = RAFDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# model
class RAFModel(nn.Module):
    def __init__(self, input_dim):
        super(RAFModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x).squeeze(1)
    
# initialize model, loss function, and optimizer
model = RAFModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training loop
num_epochs = 200
history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    
    for inputs, targets in train_loader:
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
        optimizer.step()

        # update metrics
        train_loss += loss.item()
        train_mae += torch.mean(torch.abs(outputs - targets)).item()

    # validation loop
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        val_loss = criterion(y_pred, y_test).item()
        val_mae = torch.mean(torch.abs(y_pred - y_test)).item()

    # update history
    history['loss'].append(train_loss / len(train_loader))
    history['mae'].append(train_mae / len(train_loader))
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)

    # print progress
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1}/{num_epochs}, loss: {train_loss/len(train_loader):.4f}, MAE: {train_mae/len(train_loader):.4f}, val loss: {val_loss:.4f}, val MAE: {val_mae:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'models/raf1_model_pytorch.pt')

# evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_mae = torch.mean(torch.abs(y_pred - y_test)).item()
    print(f"test MAE: {test_mae:.4f}")
    

# plot learning curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='training loss')
plt.plot(history['val_loss'], label='validation loss')
plt.title('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['mae'], label='training MAE')
plt.plot(history['val_mae'], label='validation MAE')
plt.title('mean abs error')
plt.legend()

plt.tight_layout()
plt.savefig('figures/training_history_pytorch.png')
plt.close()

# plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('actual score')
plt.ylabel('pred score')
plt.title('pred vs actual RAF1 binding fitness scores')
plt.savefig('figures/predictions_vs_actual_pytorch.png')
plt.close()

print("Feature stats:", np.min(X), np.max(X), np.mean(X), np.std(X))
print("Target stats:", np.min(y), np.max(y), np.mean(y), np.std(y))
