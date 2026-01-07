import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet

def resnet10(pretrained=False, **kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)  # [1, 1, 1, 1] 表示每个阶段的 block 数量
    if pretrained:
        raise ValueError("No pretrained model available for ResNet10")
    return model

class DynamicWeightFusion(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features))
        # self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature = 1.0  

    def forward(self, features):
        """
        input: feature list -> include n*(b,512,7,7) features
        output: fused feature -> (b,512,7,7)
        """
        weights = F.softmax(self.weights / self.temperature, dim=0)
        
        stacked = torch.stack(features, dim=0)  # (n,b,512,7,7)
        weighted = stacked * weights.view(-1,1,1,1,1)  # 广播乘法
        return weighted.sum(dim=0)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "Embedding dimension must be divisible by number of heads"

        # Define linear transformations for Query, Key, and Value
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # Output linear layer
        self.out_proj = nn.Linear(dim, dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        query, key, and value all have shape [batch_size, feature_size]
        """
        batch_size, s, feature_size = query.size()

        # Treat [batch_size, feature_size] as [batch_size, 1, feature_size], adding a seq_len dimension
        # query = query.unsqueeze(1)
        # key = key.unsqueeze(1)
        # value = value.unsqueeze(1)

        # Linear transformations, projecting to multi-head dimensions
        query = self.query_proj(query).view(batch_size, query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # (b,num_heads,key_num,d)
        value = self.value_proj(value).view(batch_size, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32, device=query.device))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)  # Add Dropout

        # Weight the Value using attention scores
        context = torch.matmul(attention_probs, value)

        # Combine multi-head results
        context = context.transpose(1, 2).contiguous().view(batch_size, s, self.dim)

        # Final linear layer to map back to the original dimension
        if s==1:
            output = self.out_proj(context).squeeze(1)  # Remove the seq_len dimension
        else:
            output = self.out_proj(context)
        return output



# class HiFusion(nn.Module):
#     def __init__(self, gene_output=250, hism_level=[1,4,49], with_cam=False):
#         super(HiFusion, self).__init__()
#         backbone = resnet18(weights=None)

#         self.spot_encoder = nn.Sequential(
#             backbone.conv1,
#             backbone.bn1,
#             backbone.relu,
#             backbone.maxpool,
#             backbone.layer1,
#             backbone.layer2,
#             backbone.layer3,
#             backbone.layer4
#         )
        
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.gene_head_250 = nn.Sequential(
#             nn.LayerNorm(512),
#             nn.Linear(512, gene_output)
#         )

#         self.with_cam = with_cam
#         if with_cam:
#             self.fc=nn.Conv2d(512, gene_output, 1, bias=False)
#         else:
#             self.fc = nn.Linear(512, gene_output)

#         self.hism_scales = hism_level

#         self.ca_avgpool = nn.AdaptiveAvgPool2d((4, 4))
#         self.ccf = CrossAttention(dim=512, num_heads=8)

#         region_backbone = resnet10()

#         self.region_encoder = nn.Sequential(
#             region_backbone.conv1,
#             region_backbone.bn1,
#             region_backbone.relu,
#             region_backbone.maxpool,
#             region_backbone.layer1,
#             region_backbone.layer2,
#             region_backbone.layer3,
#             region_backbone.layer4
#         )

#         self.fusion = DynamicWeightFusion(len(self.hism_scales))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')

#     def forward(self, region_images, spot_images):
#         batch_size = spot_images.shape[0]

#         region_feature = self.region_encoder(region_images) # (b,512,14,14)
#         query_feature = torch.flatten(self.avgpool(region_feature), 1).view(batch_size, 1, 512) # (b,1,512)

#         preds = []
#         feats = []
        
#         # HISM 
#         target_feat_size = None

#         multi_scale_feats = [spot_feature] # (b,4,512)
#         multi_scale_preds_list = []
#         rescaled_feat = []

#         for i in range(len(self.hism_scales)):
#             k=self.hism_scales[i]
#             if k==1:
#                 spot_feature = self.spot_encoder(spot_images) # (b,512,7,7)
#                 spot_pred = self.fc(torch.flatten(self.avgpool(spot_feature), 1))
#                 target_feat_size = spot_feature.shape[-1]

#                 preds.append(spot_pred)
#                 feats.append(spot_feature)
#             else:
#                 spot_tiled_images = tile_features(spot_images, k) # (b*k**2,3,224/k,224/k)
#                 spot_feature_tiled = self.spot_encoder(spot_tiled_images) # (b*k**2,512,224/k/32,224/k/32)
#                 spot_feature = merge_features(spot_feature_tiled, k, batch_size)  # (b,512,s,s)

#                 if target_feat_size == None:
#                     target_feat_size = spot_feature.shape[-1]

#                 if target_feat_size != spot_feature.shape[-1]:
#                     spot_feature = nn.AdaptiveMaxPool2d((target_feat_size, target_feat_size))(spot_feature)

#                 spot_pred = self.fc(torch.flatten(self.avgpool(spot_feature), 1))
#                 preds.append(spot_pred)
#                 feats.append(spot_feature)
        
#         # CCF
#         if len(feats)==1:
#             key_feature = torch.flatten(self.ca_avgpool(feats[0]), 2).permute(0,2,1) # (b,4,512)
#             fused_feat = self.ccf(query_feature, key_feature, key_feature)+ query_feature.squeeze(1)
#         else:
#             key_feature = self.fusion(feats)
#             key_feature = torch.flatten(self.ca_avgpool(key_feature), 2).permute(0,2,1) # (b,4,512)
#             fused_feat = self.ccf(query_feature, key_feature, key_feature)+ query_feature.squeeze(1)

#         final_pred = self.gene_head_250(fused_feat)
#         preds.append(final_pred)
       
#         return preds, feats


class HiFusion(nn.Module):
    def __init__(self, gene_output=250, hism_level=[1, 4, 49], with_cam=False):
        super().__init__()
        backbone = resnet18(weights=None)

        # Spot encoder
        self.spot_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gene_head_250 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, gene_output)
        )

        # Prediction head
        self.with_cam = with_cam
        self.fc = nn.Conv2d(512, gene_output, 1, bias=False) if with_cam else nn.Linear(512, gene_output)

        # HISM
        self.hism_scales = hism_level
        self.pool_cache = {}  

        # CCF
        self.ca_avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.ccf = CrossAttention(dim=512, num_heads=8)

        # Region encoder
        region_backbone = resnet10()
        self.region_encoder = nn.Sequential(
            region_backbone.conv1, region_backbone.bn1, region_backbone.relu,
            region_backbone.maxpool,
            region_backbone.layer1, region_backbone.layer2,
            region_backbone.layer3, region_backbone.layer4
        )

        self.fusion = DynamicWeightFusion(len(self.hism_scales))

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _adaptive_pool(self, tensor, size):
        """Cache pooling layers to avoid repeated creation in forward."""
        if size not in self.pool_cache:
            self.pool_cache[size] = nn.AdaptiveMaxPool2d((size, size))
        return self.pool_cache[size](tensor)

    def forward(self, region_images, spot_images):
        B = spot_images.shape[0]

        # Region → query feature
        region_feat = self.region_encoder(region_images)
        query_feat = self.avgpool(region_feat).flatten(1).unsqueeze(1)  # (B,1,512)

        preds, feats = [], []
        target_size = None

        # ----- HISM -----
        for k in self.hism_scales:

            if k == 1:
                # Original resolution
                spot_feat = self.spot_encoder(spot_images)  # (B,512,7,7)
            else:
                # Tiled patches
                spot_tile = tile_features(spot_images, k)  # (B*k^2,3,H/k,W/k)
                spot_feat_tile = self.spot_encoder(spot_tile)
                spot_feat = merge_features(spot_feat_tile, k, B)  # (B,512,s,s)

            # Align feature size across levels
            if target_size is None:
                target_size = spot_feat.shape[-1]
            elif spot_feat.shape[-1] != target_size:
                spot_feat = self._adaptive_pool(spot_feat, target_size)

            # Prediction for this level
            flat_feat = self.avgpool(spot_feat).flatten(1)
            spot_pred = self.fc(flat_feat) if not self.with_cam else self.fc(spot_feat).flatten(1)

            preds.append(spot_pred)
            feats.append(spot_feat)

        # ----- CCF -----
        if len(feats) == 1:
            key_feat = feats[0]
        else:
            key_feat = self.fusion(feats)

        key_feat = self.ca_avgpool(key_feat).flatten(2).transpose(1, 2)  # (B,4,512)

        fused = self.ccf(query_feat, key_feat, key_feat) + query_feat.squeeze(1)
        final_pred = self.gene_head_250(fused)

        preds.append(final_pred)
        return preds, feats