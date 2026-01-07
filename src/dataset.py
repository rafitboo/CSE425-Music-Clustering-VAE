import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MusicDataset(Dataset):
    def __init__(self, csv_file, mode='hybrid', include_label=False):
        """
        mode: 'audio', 'lyrics', or 'hybrid'
        include_label: If True, returns (features, label). If False, returns (features).
                       Set False for Easy/Medium tasks. Set True for Hard Task.
        """
        self.data = pd.read_csv(csv_file)
        self.include_label = include_label # <--- The Toggle Switch
        
        self.audio_cols = [c for c in self.data.columns if c.startswith('mfcc_')]
        self.lyric_cols = [c for c in self.data.columns if c.startswith('tfidf_')]
        
        if mode == 'audio':
            self.features = self.data[self.audio_cols].values
        elif mode == 'lyrics':
            self.features = self.data[self.lyric_cols].values
        else: 
            print(f"   [Dataset] Loading Hybrid features ({len(self.audio_cols)} Audio + {len(self.lyric_cols)} Lyric)")
            self.features = pd.concat([
                self.data[self.audio_cols], 
                self.data[self.lyric_cols]
            ], axis=1).values
            
   
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
 
        if 'genre' in self.data.columns:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.data['genre'].fillna('unknown'))
            self.num_classes = len(self.label_encoder.classes_)
        else:
            self.labels = [-1] * len(self.data)
            self.num_classes = 0


        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.include_label:
            return self.features[idx], self.labels[idx] # For Hard Task
        else:
            return self.features[idx] # For Easy/Medium Tasks

    def get_input_dim(self):
        return self.features.shape[1]