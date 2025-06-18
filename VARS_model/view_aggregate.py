import torch
import torch.nn as nn

class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # mvimages shape: [B, V, C, T, H, W]
        B, V = mvimages.shape[:2]
        
        # Reshape for batch processing
        mvimages = mvimages.view(B*V, *mvimages.shape[2:])
        
        # Extract features
        features = self.model(mvimages)
        
        # Apply lifting network if provided
        if len(self.lifting_net) > 0:
            features = self.lifting_net(features)
        
        # Reshape back to [B, V, F]
        features = features.view(B, V, -1)
        
        # Max pooling across views
        pooled_view, _ = torch.max(features, dim=1)
        
        return pooled_view, features

class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # mvimages shape: [B, V, C, T, H, W]
        B, V = mvimages.shape[:2]
        
        # Reshape for batch processing
        mvimages = mvimages.view(B*V, *mvimages.shape[2:])
        
        # Extract features
        features = self.model(mvimages)
        
        # Apply lifting network if provided
        if len(self.lifting_net) > 0:
            features = self.lifting_net(features)
        
        # Reshape back to [B, V, F]
        features = features.view(B, V, -1)
        
        # Average pooling across views
        pooled_view = torch.mean(features, dim=1)
        
        return pooled_view, features

class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        
        # Attention mechanism for view weighting
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, mvimages):
        # mvimages shape: [B, V, C, T, H, W]
        B, V = mvimages.shape[:2]
        
        # Reshape for batch processing
        mvimages = mvimages.view(B*V, *mvimages.shape[2:])
        
        # Extract features
        features = self.model(mvimages)
        
        # Apply lifting network if provided
        if len(self.lifting_net) > 0:
            features = self.lifting_net(features)
        
        # Reshape back to [B, V, F]
        features = features.view(B, V, -1)
        
        # Calculate attention weights
        attention_weights = self.attention(features)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of features
        pooled_view = torch.sum(features * attention_weights, dim=1)
        
        return pooled_view, features 