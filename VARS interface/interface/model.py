import __future__
import os, sys, torch
# Add project root to path so mvaggregate.py module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from VARS_model.mvaggregate import MVAggregate   # 7-feature extractor version
from torchvision.models.video import r2plus1d_18
from torch import nn

class MVNetwork(torch.nn.Module):
    """
    R(2+1)D based Multi-View Network for video analysis.
    Uses R(2+1)D-18 as backbone with attention-based view aggregation.
    """
    def __init__(self, net_name='r2plus1d', agr_type='attention', lifting_net=torch.nn.Sequential()):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        self.feat_dim = 512  # R(2+1)D-18 feature dimension

        # Initialize R(2+1)D-18 model
        network = r2plus1d_18(pretrained=True)
        
        # Remove the final classification layer
        network.fc = torch.nn.Sequential()

        # Initialize the aggregation model
        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=self.lifting_net,
        )

    def forward(self, mvimages):
        """
        Forward pass for R(2+1)D network.
        Args:
            mvimages: Input tensor of shape [batch_size, num_views, channels, time, height, width]
        Returns:
            Features tensor of shape [batch_size, 7]
        """
        # Handle both single tensor and list inputs
        if isinstance(mvimages, list):
            mvimages = mvimages[0]  # Use only slow path
        return self.mvnetwork(mvimages)

    def extract_features(self, mvimages):
        """
        Extract features using R(2+1)D network.
        Args:
            mvimages: Input tensor of shape [batch_size, num_views, channels, time, height, width]
        Returns:
            Features tensor of shape [batch_size, 7]
        """
        with torch.no_grad():
            return self.mvnetwork(mvimages)  # Returns single tensor
