"""Temporal Graph Neural Network (TGNN) for Behavioral Pattern Recognition

This module implements the core TGNN architecture for analyzing temporal
employee behavioral patterns and organizational dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class TGNNConfig:
    """Configuration for TGNN model"""
    node_feature_dim: int = 128
    edge_feature_dim: int = 64
    hidden_dim: int = 256
    num_gnn_layers: int = 3
    num_attention_heads: int = 8
    temporal_window: int = 30  # days
    dropout: float = 0.3
    learning_rate: float = 0.001


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for weighting time steps"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) boolean mask
        Returns:
            attended: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        Q = self.query(x)  # (batch, seq_len, hidden_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        return self.out(attended)


class GraphConvolutionLayer(nn.Module):
    """Graph convolution layer with edge features"""
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.node_conv = GATConv(in_dim, out_dim, heads=4, concat=False)
        self.edge_encoder = nn.Linear(edge_dim, in_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
        """
        if edge_attr is not None:
            # Incorporate edge features
            edge_weight = self.edge_encoder(edge_attr).mean(dim=-1)
            x = self.node_conv(x, edge_index, edge_weight)
        else:
            x = self.node_conv(x, edge_index)
        
        return self.norm(x)


class TemporalGraphEncoder(nn.Module):
    """Encodes temporal graph snapshots into node embeddings"""
    
    def __init__(self, config: TGNNConfig):
        super().__init__()
        self.config = config
        
        # Initial node feature projection
        self.node_encoder = nn.Linear(config.node_feature_dim, config.hidden_dim)
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList([
            GraphConvolutionLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.edge_feature_dim
            )
            for _ in range(config.num_gnn_layers)
        ])
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            config.hidden_dim,
            config.num_attention_heads
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, node_feature_dim)
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_feature_dim)
            timestamps: (num_nodes,) - optional temporal information
        Returns:
            embeddings: (num_nodes, hidden_dim)
        """
        # Initial encoding
        x = self.node_encoder(node_features)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply graph convolutions
        for gnn_layer in self.gnn_layers:
            x_new = gnn_layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual connection
            x = F.relu(x)
            x = self.dropout(x)
        
        return x


class BehavioralPredictionHead(nn.Module):
    """Multi-task prediction head for behavioral outcomes"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Attrition risk (binary classification)
        self.attrition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Engagement score (regression)
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Collaboration score (regression)
        self.collaboration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Anomaly detection (binary classification)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'attrition_risk': self.attrition_head(embeddings),
            'engagement_score': self.engagement_head(embeddings),
            'collaboration_score': self.collaboration_head(embeddings),
            'anomaly_likelihood': self.anomaly_head(embeddings)
        }


class TGNN(nn.Module):
    """Complete Temporal Graph Neural Network for PERSONA"""
    
    def __init__(self, config: Optional[TGNNConfig] = None):
        super().__init__()
        self.config = config or TGNNConfig()
        
        # Temporal graph encoder
        self.encoder = TemporalGraphEncoder(self.config)
        
        # Prediction heads
        self.prediction_head = BehavioralPredictionHead(self.config.hidden_dim)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TGNN
        
        Args:
            node_features: Employee feature vectors
            edge_index: Interaction graph structure
            edge_attr: Interaction features
            timestamps: Temporal information
        
        Returns:
            predictions: Dictionary of behavioral predictions
        """
        # Get node embeddings
        embeddings = self.encoder(node_features, edge_index, edge_attr, timestamps)
        
        # Get predictions
        predictions = self.prediction_head(embeddings)
        predictions['embeddings'] = embeddings
        
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Multi-task loss computation
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            weights: Task-specific loss weights
        """
        weights = weights or {
            'attrition_risk': 1.0,
            'engagement_score': 0.5,
            'collaboration_score': 0.5,
            'anomaly_likelihood': 0.8
        }
        
        total_loss = 0.0
        
        # Attrition loss (binary cross-entropy)
        if 'attrition_risk' in targets:
            attrition_loss = F.binary_cross_entropy(
                predictions['attrition_risk'].squeeze(),
                targets['attrition_risk'].float()
            )
            total_loss += weights['attrition_risk'] * attrition_loss
        
        # Engagement loss (MSE)
        if 'engagement_score' in targets:
            engagement_loss = F.mse_loss(
                predictions['engagement_score'].squeeze(),
                targets['engagement_score'].float()
            )
            total_loss += weights['engagement_score'] * engagement_loss
        
        # Collaboration loss (MSE)
        if 'collaboration_score' in targets:
            collab_loss = F.mse_loss(
                predictions['collaboration_score'].squeeze(),
                targets['collaboration_score'].float()
            )
            total_loss += weights['collaboration_score'] * collab_loss
        
        # Anomaly loss (binary cross-entropy)
        if 'anomaly_likelihood' in targets:
            anomaly_loss = F.binary_cross_entropy(
                predictions['anomaly_likelihood'].squeeze(),
                targets['anomaly_likelihood'].float()
            )
            total_loss += weights['anomaly_likelihood'] * anomaly_loss
        
        return total_loss


# Example usage
if __name__ == "__main__":
    # Create model
    config = TGNNConfig(
        node_feature_dim=128,
        hidden_dim=256,
        num_gnn_layers=3
    )
    model = TGNN(config)
    
    # Dummy data
    num_nodes = 100
    num_edges = 500
    
    node_features = torch.randn(num_nodes, config.node_feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, config.edge_feature_dim)
    
    # Forward pass
    predictions = model(node_features, edge_index, edge_attr)
    
    print("Model output shapes:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
