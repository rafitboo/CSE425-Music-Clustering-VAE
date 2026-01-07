import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


from src.dataset import MusicDataset
from src.vae import BasicVAE, loss_function
from src.evaluation import cluster_and_evaluate


FEATURE_FILE = 'final_data_features.csv'
BATCH_SIZE = 64
EPOCHS = 100        
LATENT_DIM = 8      
N_CLUSTERS = 5 

print("--- STARTING EASY TASK ---")


print(f"Loading data from {FEATURE_FILE}...")
dataset = MusicDataset(FEATURE_FILE, mode='audio', include_label=False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
input_dim = dataset.get_input_dim()
print(f"Input features per song: {input_dim}")


print("\n--- BASELINE: PCA ---")
X_full = dataset.features.numpy()

pca = PCA(n_components=LATENT_DIM)
X_pca = pca.fit_transform(X_full)


labels_pca, sil_pca, ch_pca, _, _, _, _ = cluster_and_evaluate(X_pca, n_clusters=N_CLUSTERS, algo='kmeans')
print(f"PCA Baseline -> Silhouette: {sil_pca:.4f} | Calinski-Harabasz: {ch_pca:.4f}")


print("\n--- TRAINING VAE ---")
model = BasicVAE(input_dim, latent_dim=LATENT_DIM)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    for data in dataloader: 
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Average Loss {train_loss / len(dataloader.dataset):.4f}')


print("\n--- EXTRACTING LATENT VECTORS ---")
model.eval()
with torch.no_grad():
    mu, logvar = model.encode(dataset.features)
    X_vae = mu.numpy() 


labels_vae, sil_vae, ch_vae, _, _, _, _ = cluster_and_evaluate(X_vae, n_clusters=N_CLUSTERS, algo='kmeans')
print(f"VAE Model    -> Silhouette: {sil_vae:.4f} | Calinski-Harabasz: {ch_vae:.4f}")


print("\n" + "="*40)
print(f"{'Method':<15} | {'Silhouette':<12} | {'CH Index':<12}")
print("-" * 40)
print(f"{'PCA':<15} | {sil_pca:<12.4f} | {ch_pca:<12.4f}")
print(f"{'VAE':<15} | {sil_vae:<12.4f} | {ch_vae:<12.4f}")
print("="*40)


print("\nGenerating t-SNE plot...")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_vae)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_vae, cmap='viridis', s=5, alpha=0.6)
plt.title(f"VAE Latent Space Clusters (Silhouette: {sil_vae:.2f})")
plt.colorbar(label='Cluster ID')
plt.savefig("results/latent_visualization/vae_clusters.png")
print("Saved plot to results/latent_visualization/vae_clusters.png")
