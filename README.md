
# Unsupervised Learning: VAE for Hybrid Music Clustering

**Name:** MD. Rafiul Islam  
**ID:** 22201842  
**CSE425 Neural Networks, Section-5**


## 📌 Project Overview
This project implements a deep generative pipeline to cluster music tracks using **Variational Autoencoders (VAEs)**. It explores the transition from uni-modal (Audio-only) to multi-modal (Audio + Lyrics) learning, culminating in a supervised Conditional VAE (CVAE) approach.

The project is divided into three complexity levels:
- **Easy Task:** Basic VAE on audio features (MFCCs).
- **Medium Task:** Convolutional Hybrid VAE fusing audio and lyrics (TF-IDF), tested with K-Means, Agglomerative, and DBSCAN.
- **Hard Task:** Conditional VAE (CVAE) conditioned on genre labels to disentangle the latent space.

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/rafitboo/CSE425-Music-Clustering-VAE.git
cd CSE425-Music-Clustering-VAE
```

### 2. Install Dependencies

Ensure you have Python installed. Install the required libraries using:

```bash
pip install -r requirements.txt

```

### 3. Data Setup

* **Ready-to-Run:** The pre-processed feature file **`final_data_features.csv`** is included in the root directory. You do **not** need to download the raw audio files to run the scripts.
* **Raw Data:** If you wish to access the original Audio (MP3) and Lyric files, please refer to `data/DOWNLOAD_RAW_DATA.md` for the Google Drive link.

---

## 🚀 How to Run the Project

You can reproduce the results for each task by running the corresponding script in your terminal.

### **1. Run Easy Task (Basic VAE)**

Trains a simple Linear VAE on Audio features and compares it with a PCA baseline.

```bash
python run_easy_task.py

```

* **Output:** Silhouette Score comparison (VAE vs. PCA) and a latent space plot in `results/latent_visualization/`.

### **2. Run Medium Task (Hybrid ConvVAE)**

Trains a **Convolutional VAE** on Audio + Lyrics. Evaluates clustering using K-Means, Agglomerative Clustering, and DBSCAN.

```bash
python run_medium_task.py

```

* **Output:** Clustering metrics (Silhouette, ARI, NMI) for all 3 algorithms.

### **3. Run Hard Task (Conditional VAE)**

Trains a **Conditional VAE** (CVAE) using Genre labels for supervision. Calculates **Cluster Purity** and NMI.

```bash
python run_hard_task.py

```

* **Output:** High-quality clustering metrics and a genre-colored latent space visualization.

---

## 📂 Repository Structure

```text
project/
├── data/
│   └── DOWNLOAD_RAW_DATA.md   # Link to raw Audio/Lyric datasets (Drive)
├── notebooks/
│   └── exploratory.ipynb      # EDA: Genre distribution, Audio correlations, PCA
├── results/
│   ├── clustering_metrics.csv # Summary of all experimental results
│   └── latent_visualization/  # Generated t-SNE plots of the latent space
├── src/
│   ├── vae.py                 # PyTorch VAE models (Basic, Conv, CVAE)
│   ├── dataset.py             # Custom Dataset class for loading features
│   ├── evaluation.py          # Metric calculation (Silhouette, NMI, Purity)
│   └── clustering.py          # Wrapper for K-Means, DBSCAN, Agglomerative
├── final_data_features.csv    # Pre-processed input features (Audio+Lyrics)
├── run_easy_task.py           # Entry point for Task 1
├── run_medium_task.py         # Entry point for Task 2
├── run_hard_task.py           # Entry point for Task 3
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

## 📊 Key Results Summary

| Task | Model | Algorithm | Silhouette | Davies-Bouldin ↓ | ARI | NMI | Purity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | PCA | K-Means | 0.1389 | 2.5100 | 0.0 | 0.0 | 0.0 |
| **Easy** | Basic VAE | K-Means | 0.1553 | 2.4010 | 0.0 | 0.0 | 0.0 |
| **Medium** | ConvHybrid VAE | K-Means | 0.0930 | 2.5810 | 0.0149 | 0.0849 | 0.1562 |
| **Medium** | ConvHybrid VAE | Agglomerative | 0.0738 | 2.5120 | **0.0167** | 0.0783 | 0.1557 |
| **Medium** | ConvHybrid VAE | DBSCAN | **0.2292** | 3.5470 | -0.0002 | 0.0036 | 0.1392 |
| **Hard** | **Conditional VAE** | **K-Means** | 0.2196 | **1.1650** | 0.0049 | **0.1464** | **0.1821** |

*Note: For Davies-Bouldin, lower values indicate better clustering. The Hard Task (CVAE) achieved the best structural separation (lowest DB score) and the highest semantic accuracy (Purity & NMI).*
## 📜 References

* **MTG-Jamendo Dataset:** Audio tracks.
* **MulJam Dataset:** Lyric data.

```

```
