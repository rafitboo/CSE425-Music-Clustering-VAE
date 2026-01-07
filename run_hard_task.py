import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.dataset import MusicDataset
from src.vae import ConditionalVAE, loss_function
from src.evaluation import cluster_and_evaluate

FEATURE_FILE = 'final_data_features.csv'
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 8
N_CLUSTERS = 5 

print("--- STARTING HARD TASK ---")

dataset = MusicDataset(FEATURE_FILE, mode='hybrid', include_label=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
input_dim = dataset.get_input_dim()
num_classes = dataset.num_classes

print(f"Features: {input_dim} | Classes: {num_classes}")

model = ConditionalVAE(input_dim, num_classes, latent_dim=LATENT_DIM)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Avg Loss {train_loss / len(dataloader.dataset):.4f}')

model.eval()
with torch.no_grad():
    all_features = dataset.features
    all_labels = dataset.labels
    c_onehot = torch.nn.functional.one_hot(all_labels, num_classes=num_classes).float()
    mu, logvar = model.encode(all_features, c_onehot)
    X_vae = mu.numpy()

true_genres = all_labels.numpy()


pred_labels, sil, ch, db, ari, nmi, purity = cluster_and_evaluate(
    X_vae, true_labels=true_genres, n_clusters=num_classes, algo='kmeans'
)

print("="*50)
print(f"HARD TASK RESULTS (CVAE)")
print("-" * 50)
print(f"Silhouette Score:      {sil:.4f}")
print(f"Davies-Bouldin (DB):   {db:.4f} (Lower is better)")
print(f"Adjusted Rand (ARI):   {ari:.4f}")
print(f"Norm Mutual Info(NMI): {nmi:.4f}")
print(f"Cluster Purity:        {purity:.4f}")
print("="*50)


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_vae)
plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=true_genres, cmap='tab10', s=5, alpha=0.6)
plt.title(f"CVAE Latent Space (Colored by Genre)\nPurity: {purity:.3f} | NMI: {nmi:.3f}")
plt.colorbar(label='Genre ID')
plt.savefig("results/latent_visualization/cvae_clusters.png")
print("Saved plot.")
