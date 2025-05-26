"""
Tabular Autoencoder for feature embedding in C-MAPS framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class TabularAutoencoder(nn.Module):
    """
    Autoencoder specifically designed for tabular data.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 10, hidden_dims: Optional[List[int]] = None):
        super(TabularAutoencoder, self).__init__()
        
        # Default architecture if not specified
        if hidden_dims is None:
            hidden_dims = [min(128, input_dim*4), min(64, input_dim*2), min(32, input_dim)]
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final encoder layer to embedding dimension
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final decoder layer to input dimension
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder."""
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)


class TabularAutoencoderTrainer:
    """
    Trainer class for TabularAutoencoder.
    """
    
    def __init__(self, 
                 input_dim: int,
                 embedding_dim: int = 10,
                 hidden_dims: Optional[List[int]] = None,
                 device: Optional[torch.device] = None,
                 verbose: bool = True):
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [
                min(1000, input_dim*4),
                min(500, input_dim*2),
                min(50, input_dim)
            ]
        
        # Create model
        self.model = TabularAutoencoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        if self.verbose:
            print(f"Created TabularAutoencoder on device: {self.device}")
            print(f"Input dim: {input_dim}, Embedding dim: {embedding_dim}")
            print(f"Hidden dimensions: {hidden_dims}")
    
    def train(self, 
              real_data: pd.DataFrame,
              synthetic_data: Optional[pd.DataFrame] = None,
              epochs: int = 100,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              use_synthetic_for_training: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Train the autoencoder and return embeddings.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Optional synthetic data DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            use_synthetic_for_training: Whether to include synthetic data in training
            
        Returns:
            Tuple of (real_embeddings, synthetic_embeddings)
        """
        # Convert to numpy arrays
        real_data_array = real_data.values if isinstance(real_data, pd.DataFrame) else real_data
        
        # Prepare training data
        if use_synthetic_for_training and synthetic_data is not None:
            synthetic_data_array = synthetic_data.values if isinstance(synthetic_data, pd.DataFrame) else synthetic_data
            combined_data = np.vstack([real_data_array, synthetic_data_array])
            
            # Balance datasets if synthetic data is much larger
            if len(real_data_array) < len(synthetic_data_array):
                duplication_factor = max(1, int(len(synthetic_data_array) / len(real_data_array)))
                if duplication_factor > 1:
                    real_data_duplicated = np.tile(real_data_array, (duplication_factor, 1))
                    combined_data = np.vstack([real_data_duplicated, synthetic_data_array])
        else:
            combined_data = real_data_array
        
        # Create dataset and dataloader
        train_tensor = torch.FloatTensor(combined_data)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        loss_history = []
        
        if self.verbose:
            print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_data, _ in train_loader:
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                reconstruction, _ = self.model(batch_data)
                loss = criterion(reconstruction, batch_data)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_data.size(0)
            
            epoch_loss /= len(train_loader.dataset)
            loss_history.append(epoch_loss)
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        if self.verbose:
            print(f"Training complete. Final loss: {loss_history[-1]:.6f}")
        
        # Plot loss curve
        if self.verbose:
            plt.figure(figsize=(10, 4))
            plt.plot(loss_history)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.grid(True, alpha=0.3)
            plt.show()
        
        # Generate embeddings
        real_embeddings = self._get_embeddings(real_data_array)
        
        synthetic_embeddings = None
        if synthetic_data is not None:
            synthetic_data_array = synthetic_data.values if isinstance(synthetic_data, pd.DataFrame) else synthetic_data
            synthetic_embeddings = self._get_embeddings(synthetic_data_array)
        
        return real_embeddings, synthetic_embeddings
    
    def _get_embeddings(self, data: np.ndarray) -> np.ndarray:
        """Get embeddings for given data."""
        self.model.eval()
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            _, embeddings = self.model(data_tensor)
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def get_embeddings(self, data: pd.DataFrame) -> np.ndarray:
        """Public method to get embeddings for new data."""
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        return self._get_embeddings(data_array)