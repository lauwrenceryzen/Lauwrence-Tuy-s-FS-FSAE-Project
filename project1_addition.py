# Lauwrence Tuy
# A neural network that predicts to add two numbers.

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# keep da model consistent 
SEED = 6767
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# set up da parameters 
NUM_SAMPLES = 20000     # how many pairs to generate
TRAIN_RATIO = 0.8       # 80/20 train/test
BATCH_SIZE = 128
HIDDEN_SIZE = 32        # width of hidden layers
EPOCHS = 50             # how many types theyre trained 
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# make da dataset
def make_dataset(n):
    # random # between -100 & 100
    x1 = np.random.uniform(-100.0, 100.0, size=(n, 1))
    x2 = np.random.uniform(-100.0, 100.0, size=(n, 1))
    X = np.concatenate([x1, x2], axis=1).astype(np.float32) # combines the arrays
    y = (x1 + x2).astype(np.float32)
    return X, y

X, y = make_dataset(NUM_SAMPLES)

# train/test split
split_idx = int(len(X) * TRAIN_RATIO)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test,  y_test  = X[split_idx:], y[split_idx:]

# tensor time
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test)

# create da dataset to return into pairs
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t, y_test_t)

# break da dataset into mini-batches for quicker learning times
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# time for nn archietecture
# system: 2 inputs -> hidden -> hidden -> 1 output
class SumNet(nn.Module):
    def __init__(self, in_features=2, hidden=HIDDEN_SIZE, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), # layer 1
            nn.ReLU(),
            nn.Linear(hidden, hidden), # layer 2
            nn.ReLU(),
            nn.Linear(hidden, out_features), # yay output!
        )

    def forward(self, x):
        return self.net(x)

model = SumNet().to(DEVICE) 
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop 
def train_one_epoch(epoch):
    model.train() # this signifies that this is training
    running_loss = 0.0
    for xb, yb in train_loader: # actual loop that goes thru dataset
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        # make da predictions
        preds = model(xb)
        loss = criterion(preds, yb)

        # learning time
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss # this da avg loss per epoch

# evaluate loop (similar to training loop)
def evaluate(loader):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad(): # torch.no_grad skips tracking gradient, thus memory save & saves time
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss_sum += loss.item() * xb.size(0)
    return loss_sum / len(loader.dataset)

# execute da training, spits out training info for every 5 epoch
print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    if epoch % 5 == 0 or epoch == 1:
        test_loss = evaluate(test_loader)
        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6f} | test MSE: {test_loss:.6f}")

# switch model into evaluation mode, 
model.eval()
with torch.no_grad(): # no need for gradient tracking since weve trained da model
    preds = model(X_test_t.to(DEVICE)).cpu().numpy()
true_vals = y_test
abs_err = np.abs(preds - true_vals)
within_half = (abs_err < 0.5).mean() * 100.0
within_one  = (abs_err < 1.0).mean() * 100.0
mae = abs_err.mean()

# spit out da ml stats
print("\n=== Final Test Metrics ===")
print(f"MAE (mean absolute error): {mae:.4f}")
print(f"% predictions within 0.5 of true sum: {within_half:.2f}%")
print(f"% predictions within 1.0 of true sum: {within_one:.2f}%")

# print da accurate prediction data
def demo(model, pairs):
    model.eval()
    arr = np.array(pairs, dtype=np.float32)
    with torch.no_grad():
        x = torch.from_numpy(arr).to(DEVICE)
        y_pred = model(x).cpu().numpy().reshape(-1)
    for (a, b), p in zip(pairs, y_pred):
        print(f"Input: ({a:.2f}, {b:.2f}) | Pred: {p:.2f} | Actual: {(a+b):.2f}")

print("\nPredictions:")
demo_pairs = [
    (3.0, 5.0),
    (-12.5, 7.25),
    (42.0, -10.0),
    (99.9, 0.1),
    (-55.0, -45.0),
]
demo(model, demo_pairs)
