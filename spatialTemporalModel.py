import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class SpatialTemporalDataset(Dataset):
    """Dataset class for spatial-temporal prediction."""

    def __init__(self, data: np.ndarray, neighbor_dict, spatial_features, n_temporal=5, k_neighbors=5):
        self.data = data
        self.neighbor_dict = neighbor_dict
        self.spatial_features = spatial_features
        self.n_temporal = n_temporal
        self.k_neighbors = k_neighbors
        self.n_sensors = data.shape[1]
        self.n_timesteps = data.shape[0]
        self.samples = [(t, i) for t in range(n_temporal, self.n_timesteps) for i in range(self.n_sensors)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, sensor_idx = self.samples[idx]
        target = self.data[t, sensor_idx]
        spatial_feat = self.spatial_features[sensor_idx].flatten()
        neighbors = self.neighbor_dict[sensor_idx]
        temporal_feat = []

        # target sensor history
        for lag in range(1, self.n_temporal + 1):
            temporal_feat.append(self.data[t - lag, sensor_idx])

        # neighbors’ history
        for nbr in neighbors:
            for lag in range(self.n_temporal + 1):
                temporal_feat.append(self.data[t - lag, nbr])

        features = np.concatenate([spatial_feat, temporal_feat])
        return torch.FloatTensor(features), torch.FloatTensor([target])


class SpatialTemporalPredictor(nn.Module):
    """Fully connected network for spatial-temporal prediction."""

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.2, activation="relu"):
        super().__init__()
        layers = []
        prev_dim = input_dim

        act_fn = nn.ReLU() if activation.lower() == "relu" else nn.Tanh()

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SpatialTemporalTrainer:
    """Trainer for spatial-temporal model."""

    def __init__(self, pair_id, config, train_data, init_data=None, prediction_timesteps=None, delta_t=None):
        self.pair_id = pair_id
        self.config = config
        self.train_data = np.array(train_data[0])
        self.init_data = init_data
        self.prediction_timesteps = prediction_timesteps
        self.delta_t = delta_t

        self.k_neighbors = config['model'].get('k_neighbors', 5)
        self.n_temporal = config['model'].get('n_temporal', 5)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scaler = StandardScaler()
        self.neighbor_dict = None
        self.spatial_features = None
        self.model = None

        self._compute_neighbors()
        self._compute_spatial_features()
        self._prepare_datasets()
        self.build_model()

    def _compute_neighbors(self):
        n_sensors = self.train_data.shape[1]
        coords = np.arange(n_sensors).reshape(-1, 1)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        self.neighbor_dict = {i: indices[i][1:].tolist() for i in range(n_sensors)}

    def _compute_spatial_features(self):
        n_sensors = self.train_data.shape[1]
        coords = np.arange(n_sensors).reshape(-1, 1)
        self.spatial_features = np.zeros((n_sensors, self.k_neighbors, coords.shape[1]))
        for i in range(n_sensors):
            for j, nbr in enumerate(self.neighbor_dict[i]):
                self.spatial_features[i, j] = coords[i] - coords[nbr]

    def _prepare_datasets(self, train_ratio=0.8):
        data_norm = self.scaler.fit_transform(self.train_data.T).T
        n_timesteps = data_norm.shape[0]
        train_size = int(n_timesteps * train_ratio)
        train_data, val_data = data_norm[:train_size], data_norm[train_size:]

        self.train_dataset = SpatialTemporalDataset(train_data, self.neighbor_dict, self.spatial_features,
                                                    self.n_temporal, self.k_neighbors)
        self.val_dataset = SpatialTemporalDataset(val_data, self.neighbor_dict, self.spatial_features,
                                                  self.n_temporal, self.k_neighbors)

        # input dimension matches __getitem__
        k = self.k_neighbors
        self.input_dim = self.k_neighbors*(self.n_temporal + 2) + self.n_temporal

    def build_model(self):
        hidden_layers = self.config['model'].get('hidden_layers', 3)
        neurons = self.config['model'].get('neurons', 128)
        hidden_dims = [neurons] * hidden_layers
        self.model = SpatialTemporalPredictor(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=self.config['model'].get('dropout', 0.2)
        ).to(self.device)

    def train(self):
        batch_size = self.config['model'].get('batch_size', 64)
        lr = self.config['model'].get('learning_rate', 1e-3)
        epochs = self.config['model'].get('epochs', 100)

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X), y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

        # validation loss
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                val_losses.append(criterion(self.model(X), y).item())
        self.val_loss = np.mean(val_losses)
        print(f"Validation Loss: {self.val_loss:.6f}")

    def predict(self):
        horizon = len(self.prediction_timesteps) if self.prediction_timesteps is not None else self.train_data.shape[0]
        n_sensors = self.train_data.shape[1]
        preds = np.zeros((horizon, n_sensors))

        data = self.scaler.transform(self.train_data.T).T
        window = data[-self.n_temporal:]

        for t in range(horizon):
            for i in range(n_sensors):
                spatial_feat = self.spatial_features[i].flatten()
                neighbors = self.neighbor_dict[i]
                temporal_feat = []
                for lag in range(1, self.n_temporal + 1):
                    temporal_feat.append(window[-lag, i])
                for nbr in neighbors:
                    for lag in range(self.n_temporal + 1):
                        temporal_feat.append(window[-lag, nbr])
                feat = np.concatenate([spatial_feat, temporal_feat])
                feat = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model(feat).cpu().numpy().item()
                preds[t, i] = pred
            # update window
            new_row = preds[t, :]
            window = np.vstack([window[1:], new_row])

        # inverse transform
        preds = self.scaler.inverse_transform(preds.T).T
        return preds