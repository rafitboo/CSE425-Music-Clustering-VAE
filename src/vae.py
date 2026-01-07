import torch
import torch.nn as nn
import torch.nn.functional as F

# EASY TASK (Basic VAE)
class BasicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(BasicVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mean = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# MEDIUM TASK (Convolutional Hybrid VAE)
class ConvHybridVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(ConvHybridVAE, self).__init__()
        
        # Encoder: 1D Convolution 
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        

        self.flatten_size = input_dim // 4 * 64 
        if input_dim % 4 != 0: self.flatten_size = (input_dim // 4 + 1) * 64 # adjust for padding

        # Latent Vectors
        self.fc_mean = nn.Linear(1024, latent_dim) 

        
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=0)
        

        self.fc_final = nn.Linear(16, input_dim) 

    def encode(self, x):

        x = x.unsqueeze(1) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        

        if x.shape[1] != 1024:
            self.adapter = nn.Linear(x.shape[1], 1024).to(x.device)
            x = self.adapter(x)

        return self.fc_mean(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(x.size(0), 64, 16) # Reshape back for Deconv
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        x = x.squeeze(1)
        

        if x.shape[1] != 63: 
            
            x = F.interpolate(x.unsqueeze(1), size=63, mode='linear', align_corners=False).squeeze(1)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# HARD TASK (Conditional VAE)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, num_classes, latent_dim=8):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_dim + num_classes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mean = nn.Linear(64, latent_dim)
        self.fc3_logvar = nn.Linear(64, latent_dim)
        
        self.fc4 = nn.Linear(latent_dim + num_classes, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)
        
        self.dropout = nn.Dropout(0.2)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        return self.fc3_mean(h2), self.fc3_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h4 = F.relu(self.fc4(inputs))
        h5 = F.relu(self.fc5(h4))
        h5 = self.dropout(h5)
        return self.fc6(h5)

    def forward(self, x, c):
        c = F.one_hot(c, num_classes=self.num_classes).float()
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD