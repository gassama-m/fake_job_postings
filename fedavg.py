import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time


data = pd.read_excel("/content/drive/MyDrive/fake_job_postings/preprocessed_data.xlsx")
data = data.iloc[:, [i for i in range(11, 22) if i != 17]]

X = data.drop(columns=['fraudulent'])
y = data['fraudulent'].values

categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)



X = pd.get_dummies(X, columns=categorical_cols)


scaler = StandardScaler()
X = scaler.fit_transform(X)


def create_non_iid_splits(X, y, num_clients=5, alpha=0.01, random_state=42):
    np.random.seed(random_state)
    while True:
        client_indices = [[] for _ in range(num_clients)]
        labels = np.unique(y)
        for lbl in labels:
            idx = np.where(y == lbl)[0]
            np.random.shuffle(idx)
            proportions = np.random.dirichlet([alpha]*num_clients)
            counts = (proportions * len(idx)).astype(int)
            counts[-1] = len(idx) - sum(counts[:-1])
            start = 0
            for i, c in enumerate(counts):
                client_indices[i].extend(idx[start:start+c])
                start += c
        if all(len(inds) > 0 for inds in client_indices):
            break
    client_datasets = [(X[inds], y[inds]) for inds in client_indices]
    return client_datasets

num_clients = 5
client_datasets = create_non_iid_splits(X, y, num_clients=num_clients, alpha=0.05)


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_fedavg(client_datasets, global_model, rounds=50, epochs=5, lr=0.01, batch_size=32, patience=3):
    global_weights = global_model.state_dict()
    best_acc = 0
    rounds_no_improve = 0
    total_params = sum(p.numel() for p in global_model.parameters())

    total_start = time.time()

    for r in range(rounds):
        client_weights = []
        client_sizes = []

        for Xc, yc in client_datasets:
            model = MLP(input_dim=X.shape[1])
            model.load_state_dict(global_weights)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # Convert to TensorDataset and DataLoader
            dataset = TensorDataset(torch.tensor(Xc, dtype=torch.float32),
                                    torch.tensor(yc, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Local training
            for e in range(epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()

            client_weights.append(model.state_dict())
            client_sizes.append(len(yc))

       
        new_global_weights = {}
        for key in global_weights.keys():
            new_global_weights[key] = sum(client_weights[i][key]*client_sizes[i]
                                          for i in range(len(client_datasets))) / sum(client_sizes)
        global_weights = new_global_weights
        global_model.load_state_dict(global_weights)

       
        with torch.no_grad():
            outputs = global_model(torch.tensor(X, dtype=torch.float32))
            preds = torch.argmax(outputs, dim=1).numpy()
            acc = accuracy_score(y, preds)

        print(f"Round {r+1}/{rounds} completed | Accuracy: {acc*100:.2f}%")

        # Early stopping logic
        if acc > best_acc:
            best_acc = acc
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        if rounds_no_improve >= patience:
            print(f"Early stopping triggered at round {r+1}")
            break

    total_time = time.time() - total_start

    # Communication cost 
    model_size_MB = total_params * 4 / (1024**2)  # float32 = 4 bytes
    total_comm_MB = 2 * num_clients * (r+1) * model_size_MB

    print(f"\nFedAvg Global Model Accuracy: {best_acc*100:.2f}%")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Total Rounds (Convergence): {r+1}")
    print(f"Model Size: {total_params} parameters (~{model_size_MB:.4f} MB)")
    print(f"Total Communication Cost (send + receive): ~{total_comm_MB:.2f} MB")

    return best_acc, r+1, total_time, total_comm_MB


global_model = MLP(input_dim=X.shape[1])
best_acc, total_rounds, total_time, total_comm_MB = train_fedavg(
    client_datasets, global_model, rounds=20, epochs=50, lr=0.02, batch_size=64, patience=3
)
