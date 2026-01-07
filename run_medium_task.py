import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from src.dataset import MusicDataset
from src.vae import ConvHybridVAE, loss_function
from src.evaluation import cluster_and_evaluate


FEATURE_FILE = 'final_data_features.csv'
BATCH_SIZE = 64
EPOCHS = 80         
LATENT_DIM = 16
N_CLUSTERS = 5 

print("--- STARTING MEDIUM TASK ---")


dataset = MusicDataset(FEATURE_FILE, mode='hybrid', include_label=True) 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
input_dim = dataset.get_input_dim()


df_full = pd.read_csv(FEATURE_FILE)
true_genres = dataset.labels.numpy() 
print(f"Loaded {len(true_genres)} samples.")


print("\n--- TRAINING CONV VAE ---")
model = ConvHybridVAE(input_dim, latent_dim=LATENT_DIM)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for data, _ in dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Avg Loss {train_loss / len(dataloader.dataset):.4f}')


print("\n--- EVALUATING CLUSTERING ALGORITHMS ---")
model.eval()
with torch.no_grad():
    mu, logvar = model.encode(dataset.features)
    X_vae = mu.numpy()


algos = ['kmeans', 'agglomerative', 'dbscan']

print(f"{'Algo':<15} | {'Sil':<7} | {'DB (Low=Good)':<15} | {'ARI':<7} | {'NMI':<7} | {'Purity':<7}")
print("-" * 75)

for algo in algos:
    
    pred, sil, ch, db, ari, nmi, purity = cluster_and_evaluate(
        X_vae, true_labels=true_genres, n_clusters=N_CLUSTERS, algo=algo
    )
    print(f"{algo:<15} | {sil:<7.4f} | {db:<15.4f} | {ari:<7.4f} | {nmi:<7.4f} | {purity:<7.4f}")


print("\nGenerating Plot (K-Means)...")

pred_k, _, _, _, _, _, _ = cluster_and_evaluate(X_vae, n_clusters=N_CLUSTERS, algo='kmeans')

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_vae)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=pred_k, cmap='coolwarm', s=5, alpha=0.6)
plt.title(f"Conv-Hybrid VAE Clusters (K-Means)")
plt.colorbar(label='Cluster')
os.makedirs("results/latent_visualization", exist_ok=True)
plt.savefig("results/latent_visualization/hybrid_vae_clusters.png")
print("Saved plot to results/latent_visualization/hybrid_vae_clusters.png")
