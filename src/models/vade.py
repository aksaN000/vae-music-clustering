"""
Variational Deep Embedding (VaDE)
VAE with GMM priors for joint clustering and representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
from sklearn.mixture import GaussianMixture


class VaDE(nn.Module):
    """
    Variational Deep Embedding
    Combines VAE with Gaussian Mixture Model for clustering
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        n_clusters: int = 15,
        hidden_dims: list = [512, 256],
        dropout: float = 0.2
    ):
        super(VaDE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims))
        
        in_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # GMM parameters (learnable)
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)  # Mixture weights
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))  # Cluster means
        self.logvar_c = nn.Parameter(torch.randn(n_clusters, latent_dim))  # Cluster variances
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space"""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        
        return recon, mu, logvar, z
    
    def get_gamma(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute cluster assignment probabilities
        p(c|z) for each cluster c
        """
        batch_size = z.size(0)
        
        # Normalize pi
        pi = F.softmax(self.pi, dim=0)
        
        # Compute log p(z|c) for each cluster
        log_p_z_c = []
        for k in range(self.n_clusters):
            # Log probability under Gaussian
            mu_k = self.mu_c[k]
            logvar_k = self.logvar_c[k]
            var_k = torch.exp(logvar_k)
            
            # (z - mu_k)^2 / var_k
            diff = z - mu_k.unsqueeze(0)
            log_p = -0.5 * torch.sum(
                logvar_k + (diff ** 2) / var_k + np.log(2 * np.pi),
                dim=1
            )
            log_p_z_c.append(log_p)
        
        log_p_z_c = torch.stack(log_p_z_c, dim=1)  # (batch_size, n_clusters)
        
        # Add log pi
        log_pi = torch.log(pi + 1e-10)
        log_p_c_z = log_p_z_c + log_pi.unsqueeze(0)
        
        # Normalize (softmax in log space)
        gamma = F.softmax(log_p_c_z, dim=1)
        
        return gamma
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster assignments"""
        mu, _ = self.encode(x)
        gamma = self.get_gamma(mu)
        return torch.argmax(gamma, dim=1)
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation"""
        mu, _ = self.encode(x)
        return mu
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        VaDE loss function
        L = reconstruction loss + KL(q(z|x) || p(z))
        where p(z) is GMM prior
        """
        # Flatten both to match dimensions
        batch_size = recon_x.size(0)
        recon_x_flat = recon_x.view(batch_size, -1)
        x_flat = x.view(batch_size, -1)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x_flat, x_flat, reduction='mean')
        
        # Get cluster assignment probabilities
        gamma = self.get_gamma(z)
        
        # Normalize pi
        pi = F.softmax(self.pi, dim=0)
        
        # Compute KL divergence with GMM prior
        # KL = E_gamma[log q(z|x) - log p(z|c) - log p(c)]
        
        # log q(z|x)
        log_q_z_x = -0.5 * torch.sum(
            logvar + ((z - mu) ** 2) / torch.exp(logvar) + np.log(2 * np.pi),
            dim=1
        )
        
        # log p(z|c) for each cluster
        log_p_z_c = []
        for k in range(self.n_clusters):
            mu_k = self.mu_c[k]
            logvar_k = self.logvar_c[k]
            var_k = torch.exp(logvar_k)
            
            diff = z - mu_k.unsqueeze(0)
            log_p = -0.5 * torch.sum(
                logvar_k + (diff ** 2) / var_k + np.log(2 * np.pi),
                dim=1
            )
            log_p_z_c.append(log_p)
        
        log_p_z_c = torch.stack(log_p_z_c, dim=1)
        
        # log p(c)
        log_pi = torch.log(pi + 1e-10)
        
        # Weighted sum
        kl_loss = torch.mean(
            torch.sum(gamma * (log_q_z_x.unsqueeze(1) - log_p_z_c - log_pi.unsqueeze(0)), dim=1)
        )
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def pretrain(self, data_loader, device, epochs=10):
        """
        Pre-train with standard VAE, then initialize GMM
        """
        # Move model to device
        self.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        print("Pre-training VAE...")
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                if isinstance(batch, dict):
                    x = batch['features']
                else:
                    x = batch
                
                x = x.to(device)
                
                # Forward
                recon, mu, logvar, z = self(x)
                
                # Simple VAE loss - flatten both to match
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)
                recon_flat = recon.view(batch_size, -1)
                
                recon_loss = F.mse_loss(recon_flat, x_flat)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")
        
        # Initialize GMM parameters
        print("\nInitializing GMM parameters...")
        self.eval()
        latents = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    x = batch['features']
                else:
                    x = batch
                
                x = x.to(device)
                mu, _ = self.encode(x)
                latents.append(mu.cpu().numpy())
        
        latents = np.concatenate(latents, axis=0)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
        gmm.fit(latents)
        
        # Initialize parameters
        self.pi.data = torch.from_numpy(gmm.weights_).float().to(device)
        self.mu_c.data = torch.from_numpy(gmm.means_).float().to(device)
        self.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float().to(device))
        
        print("Pre-training complete!")


if __name__ == "__main__":
    print("Testing VaDE:")
    
    input_dim = 128 * 1293
    model = VaDE(
        input_dim=input_dim,
        latent_dim=128,
        n_clusters=15,
        hidden_dims=[512, 256]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 8
    x = torch.randn(batch_size, input_dim)
    
    recon, mu, logvar = model(x)
    z = model.reparameterize(mu, logvar)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
    
    # Test cluster assignment
    gamma = model.get_gamma(z)
    print(f"Gamma shape: {gamma.shape}")
    print(f"Cluster probabilities (first sample): {gamma[0]}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"Predicted clusters: {predictions}")
    
    # Test loss
    loss_dict = model.loss_function(recon, x, mu, logvar, z)
    print(f"\nTotal loss: {loss_dict['loss'].item():.4f}")
    print(f"Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"KL loss: {loss_dict['kl_loss'].item():.4f}")
