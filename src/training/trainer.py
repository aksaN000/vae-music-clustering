"""
Training Script with GPU Optimization
Supports multi-GPU training, mixed precision, and 15-core data loading
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Dict, Optional
import time


class VAETrainer:
    """Trainer for VAE models with GPU optimization"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        save_dir: str = 'results/checkpoints',
        condition_type: str = None,  # 'language' or 'genre' for ConditionalVAE
        kl_anneal_epochs: int = 20  # Gradually increase KL weight over epochs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.condition_type = condition_type  # Store for CVAE training
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_beta = 0.0  # Start with beta=0, anneal to 1.0
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_recon_loss': [],
            'train_kl_loss': [],
            'val_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            condition = None
            
            if 'features' in batch:
                x = batch['features'].to(self.device, non_blocking=True)
            elif 'audio_features' in batch:
                x = batch['audio_features'].to(self.device, non_blocking=True)
            else:
                continue
            
            # Check if model needs condition (ConditionalVAE)
            if hasattr(self.model, 'forward') and 'condition' in self.model.forward.__code__.co_varnames:
                # Use appropriate condition type
                if self.condition_type == 'genre' and 'genre' in batch:
                    condition = batch['genre'].to(self.device, non_blocking=True)
                elif self.condition_type == 'language' and 'language' in batch:
                    condition = batch['language'].to(self.device, non_blocking=True)
                elif 'language' in batch:  # Default to language
                    condition = batch['language'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast(device_type='cuda', dtype=torch.float16):
                    if condition is not None:
                        model_output = self.model(x, condition)
                    else:
                        model_output = self.model(x)
                    
                    # Handle different return formats
                    if len(model_output) == 4:
                        recon, mu, logvar, z = model_output
                    else:
                        recon, mu, logvar = model_output
                        z = None
                    
                    # Compute loss with annealed beta
                    if hasattr(self.model, 'loss_function'):
                        import inspect
                        sig = inspect.signature(self.model.loss_function)
                        if 'z' in sig.parameters and z is not None:
                            loss_dict = self.model.loss_function(recon, x, mu, logvar, z)
                        elif 'beta' in sig.parameters:
                            loss_dict = self.model.loss_function(recon, x, mu, logvar, beta=self.current_beta)
                        else:
                            loss_dict = self.model.loss_function(recon, x, mu, logvar)
                    else:
                        from src.models.vae import vae_loss
                        loss_dict = vae_loss(recon, x, mu, logvar, beta=self.current_beta)
                    
                    loss = loss_dict['loss']
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard training
                if condition is not None:
                    model_output = self.model(x, condition)
                else:
                    model_output = self.model(x)
                
                # Handle different return formats
                if len(model_output) == 4:
                    recon, mu, logvar, z = model_output
                else:
                    recon, mu, logvar = model_output
                    z = None
                
                if hasattr(self.model, 'loss_function'):
                    import inspect
                    sig = inspect.signature(self.model.loss_function)
                    if 'z' in sig.parameters and z is not None:
                        loss_dict = self.model.loss_function(recon, x, mu, logvar, z)
                    else:
                        loss_dict = self.model.loss_function(recon, x, mu, logvar)
                else:
                    from src.models.vae import vae_loss
                    loss_dict = vae_loss(recon, x, mu, logvar)
                
                loss = loss_dict['loss']
                
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{loss_dict['recon_loss'].item():.4f}",
                    'kl': f"{loss_dict['kl_loss'].item():.4f}"
                })
        
        # Average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            condition = None
            
            if 'features' in batch:
                x = batch['features'].to(self.device, non_blocking=True)
            elif 'audio_features' in batch:
                x = batch['audio_features'].to(self.device, non_blocking=True)
            else:
                continue
            
            # Check if model needs condition (ConditionalVAE)
            if hasattr(self.model, 'forward') and 'condition' in self.model.forward.__code__.co_varnames:
                # Use appropriate condition type
                if self.condition_type == 'genre' and 'genre' in batch:
                    condition = batch['genre'].to(self.device, non_blocking=True)
                elif self.condition_type == 'language' and 'language' in batch:
                    condition = batch['language'].to(self.device, non_blocking=True)
                elif 'language' in batch:  # Default to language
                    condition = batch['language'].to(self.device, non_blocking=True)
            
            # Forward
            if condition is not None:
                model_output = self.model(x, condition)
            else:
                model_output = self.model(x)
            
            # Handle different return formats
            if len(model_output) == 4:
                recon, mu, logvar, z = model_output
            else:
                recon, mu, logvar = model_output
                z = None
            
            # Compute loss with annealed KL weight
            if hasattr(self.model, 'loss_function'):
                import inspect
                sig = inspect.signature(self.model.loss_function)
                if 'z' in sig.parameters and z is not None:
                    loss_dict = self.model.loss_function(recon, x, mu, logvar, z)
                elif 'beta' in sig.parameters:
                    loss_dict = self.model.loss_function(recon, x, mu, logvar, beta=self.current_beta)
                else:
                    loss_dict = self.model.loss_function(recon, x, mu, logvar)
            else:
                from src.models.vae import vae_loss
                loss_dict = vae_loss(recon, x, mu, logvar, beta=self.current_beta)
            
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        latest_path = self.save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model at epoch {epoch}")
        
        # Save periodic
        if epoch % 10 == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, early_stopping_patience: int = 15):
        """Full training loop"""
        print(f"\nTraining for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"KL Annealing: 0.0 -> 1.0 over {self.kl_anneal_epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Update beta for KL annealing
            if epoch <= self.kl_anneal_epochs:
                self.current_beta = epoch / self.kl_anneal_epochs
            else:
                self.current_beta = 1.0
            
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_kl_loss'].append(train_metrics['kl_loss'])
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_kl_loss'].append(val_metrics['kl_loss'])
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Print metrics
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s) [Beta: {self.current_beta:.3f}]")
                print(f"  Train Loss: {train_metrics['loss']:.4f} "
                      f"(Recon: {train_metrics['recon_loss']:.4f}, "
                      f"KL: {train_metrics['kl_loss']:.4f})")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                      f"(Recon: {val_metrics['recon_loss']:.4f}, "
                      f"KL: {val_metrics['kl_loss']:.4f})")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
            else:
                # No validation, just save periodically
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history


if __name__ == "__main__":
    print("Trainer module - use train_vae.py to run training")
