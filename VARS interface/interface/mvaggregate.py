from interface.utils import batch_tensor, unbatch_tensor
import torch
from torch import nn


class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        num_heads = 8
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        
        
        self.relu = nn.ReLU()

    def forward(self, mvimages):
        # Handle both tensor and list inputs
        slow_path = mvimages if isinstance(mvimages, torch.Tensor) else mvimages[0]
        B, V, C, T, H, W = slow_path.shape  # Batch, Views, Channel, Time, Height, Width
        
        # Print shapes for debugging
        print("Input shape:", slow_path.shape)
        
        # Process through model
        # First permute to get correct dimension order [B, V, C, T, H, W] -> [B, V, T, C, H, W]
        slow_path = slow_path.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        print("After permute:", slow_path.shape)
        
        # Calculate total elements to verify reshape
        total = slow_path.numel()
        print("Total elements:", total)
        
        # Then reshape for model input
        # [B, V, T, C, H, W] -> [B*V, C, T, H, W]
        slow_path = slow_path.reshape(B*V, C, T, H, W)
        
        print("Reshaped for model:", slow_path.shape)
        
        # Process through model - only use slow path
        features = self.model(slow_path)
        
        print("Model output shape:", features.shape)
        
        # Process through lifting net
        aux = self.lifting_net(features)
        print("After lifting net shape:", aux.shape)
        
        # Reshape back to [B, V, C]
        aux = aux.reshape(B, V, -1)
        print("Final aux shape:", aux.shape)

        # Apply attention mechanism
        aux = torch.matmul(aux, self.attention_weights)
        aux_t = aux.permute(0, 2, 1)
        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T
        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))
        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))
        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # Handle SlowFast input format
        slow_path, fast_path = mvimages
        B, V, C, D, H, W = slow_path.shape  # Batch, Views, Channel, Depth, Height, Width
        
        # Process through SlowFast model
        features = self.model([slow_path.squeeze(1), fast_path.squeeze(1)])
        aux = self.lifting_net(features)
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        # Handle SlowFast input format
        slow_path, fast_path = mvimages
        B, V, C, D, H, W = slow_path.shape  # Batch, Views, Channel, Depth, Height, Width
        
        # Process through SlowFast model
        features = self.model([slow_path.squeeze(1), fast_path.squeeze(1)])
        aux = self.lifting_net(features)
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential()):
        super().__init__()
        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net)

    def forward(self, mvimages):
        pooled_view, attention = self.aggregation_model(mvimages)
        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)
        return pred_offence_severity, pred_action, attention
