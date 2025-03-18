import torch
from torch import nn
import math

# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings


# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=8): #this impacts its power
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
#         self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze step: Global Average Pooling (GAP)
#         avg_pool = torch.mean(x, dim=[2, 3, 4], keepdim=True)  # (batch_size, channels, 1, 1, 1)

#         # Excitation step: Fully connected layers
#         excitation = self.fc1(avg_pool)  # (batch_size, channels // reduction, 1, 1, 1)
#         excitation = nn.ReLU()(excitation)
#         excitation = self.fc2(excitation)  # (batch_size, channels, 1, 1, 1)
#         excitation = self.sigmoid(excitation)

#         # Scale the input with attention weights
#         return x * excitation

# class Network(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_channels=32, time_emb_dim = 32):
#         super(Network, self).__init__()
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.ReLU()
#         )
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(base_channels),
#             nn.ReLU()
#         )
#         self.se1 = SEBlock(base_channels)  # SE block after encoder1

#         self.encoder2 = nn.Sequential(
#             nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 2),
#             nn.ReLU()
#         )
#         self.se2 = SEBlock(base_channels * 2)  # SE block after encoder2

#         self.encoder3 = nn.Sequential(
#             nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 4),
#             nn.ReLU()
#         )
#         self.se3 = SEBlock(base_channels * 4)  # SE block after encoder3

#         self.encoder4 = nn.Sequential(
#             nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 8),
#             nn.ReLU()
#         )
#         self.se4 = SEBlock(base_channels * 8)  # SE block after encoder4

#         self.encoder5 = nn.Sequential(
#             nn.Conv3d(base_channels * 8, base_channels * 16, kernel_size=3, stride=(1, 2, 2), padding=1),
#             nn.BatchNorm3d(base_channels * 16),
#             nn.ReLU()
#         )
#         self.se5 = SEBlock(base_channels * 16)  # SE block after encoder5

#         # Decoder
#         self.decoder5 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
#             nn.BatchNorm3d(base_channels * 8),
#             nn.ReLU()
#         )
#         self.decoder4 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels * 4),
#             nn.ReLU()
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels * 2),
#             nn.ReLU()
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels),
#             nn.ReLU()
#         )
#         self.decoder1 = nn.Sequential(
#             nn.Conv3d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # To scale output to [0, 1]
#         )

#     def forward(self, x, time = 11):
        
#         time = torch.tensor([time], dtype=torch.float32, device='cpu')
#         time_emb = self.time_mlp(time)
#         time_emb = time_emb[:, :, None, None, None] 

#         skip1 = self.encoder1(x) + time_emb
#         # print('shape ', skip1.shape, time_emb.shape)
#         skip1 = self.se1(skip1)

#         skip2 = self.encoder2(skip1)
#         time_emb = time_emb.repeat(1, 2, 1, 1, 1)  # Expands 32 → 64 by repeating
#         skip2 = self.se2(skip2 + time_emb)
#         skip3 = self.encoder3(skip2) 
#         time_emb = time_emb.repeat(1, 2, 1, 1, 1)  # Expands 32 → 64 by repeating
#         skip3 = self.se3(skip3 + time_emb)
#         skip4 = self.encoder4(skip3)
#         time_emb = time_emb.repeat(1, 2, 1, 1, 1)  # Expands 32 → 64 by repeating
#         skip4 = self.se4(skip4 + time_emb)
#         encoded = self.encoder5(skip4)
#         time_emb = time_emb.repeat(1, 2, 1, 1, 1)  # Expands 32 → 64 by repeating
#         encoded = self.se5(encoded + time_emb)

#         # Decode
#         decoded = self.decoder5(encoded)
#         decoded = self.decoder4(decoded)
#         decoded = self.decoder3(decoded)
#         decoded = self.decoder2(decoded)
#         output = self.decoder1(decoded )

#         return output


# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=8): #this impacts its power
#         super(SEBlock, self).__init__()
#         self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
#         self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Squeeze step: Global Average Pooling (GAP)
#         avg_pool = torch.mean(x, dim=[2, 3, 4], keepdim=True)  # (batch_size, channels, 1, 1, 1)

#         # Excitation step: Fully connected layers
#         excitation = self.fc1(avg_pool)  # (batch_size, channels // reduction, 1, 1, 1)
#         excitation = nn.ReLU()(excitation)
#         excitation = self.fc2(excitation)  # (batch_size, channels, 1, 1, 1)
#         excitation = self.sigmoid(excitation)

#         # Scale the input with attention weights
#         return x * excitation



# =========================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(Network, self).__init__()

        # Encoder Layers
        self.encoder1 = self._conv_block(in_channels, base_channels)
        self.encoder2 = self._conv_block(base_channels, base_channels * 2, stride=2)
        self.encoder3 = self._conv_block(base_channels * 2, base_channels * 4, stride=2)
        self.encoder4 = self._conv_block(base_channels * 4, base_channels * 8, stride=2)
        self.encoder5 = self._conv_block(base_channels * 8, base_channels * 16, stride=2)
        self.encoder6 = self._conv_block(base_channels * 16, base_channels * 32, stride=2)
        self.encoder7 = self._conv_block(base_channels * 32, base_channels * 64, stride=2)

        # Decoder Layers with Skip Connections
        self.decoder7 = self._deconv_block(base_channels * 64, base_channels * 32)
        self.decoder6 = self._deconv_block(base_channels * 64, base_channels * 16)
        self.decoder5 = self._deconv_block(base_channels * 32, base_channels * 8)
        self.decoder4 = self._deconv_block(base_channels * 16, base_channels * 4)
        self.decoder3 = self._deconv_block(base_channels * 8, base_channels * 2)
        self.decoder2 = self._deconv_block(base_channels * 4, base_channels)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(base_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder pathway
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x6 = self.encoder6(x5)
        x7 = self.encoder7(x6)

        # Decoder pathway with skip connections
        d7 = self.decoder7(x7)
        d7 = self._pad_if_needed(d7, x6)
        d7 = torch.cat([d7, x6], dim=1)

        d6 = self.decoder6(d7)
        d6 = self._pad_if_needed(d6, x5)
        d6 = torch.cat([d6, x5], dim=1)

        d5 = self.decoder5(d6)
        d5 = self._pad_if_needed(d5, x4)
        d5 = torch.cat([d5, x4], dim=1)

        d4 = self.decoder4(d5)
        d4 = self._pad_if_needed(d4, x3)
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.decoder3(d4)
        d3 = self._pad_if_needed(d3, x2)
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.decoder2(d3)
        d2 = self._pad_if_needed(d2, x1)
        d2 = torch.cat([d2, x1], dim=1)

        # Final output layer
        out = self.decoder1(d2)
        return out

    def _pad_if_needed(self, x, target):
        """Pad tensor x to match the shape of target if necessary."""
        diff_depth = target.shape[2] - x.shape[2]
        diff_height = target.shape[3] - x.shape[3]
        diff_width = target.shape[4] - x.shape[4]
        x = F.pad(x, [0, diff_width, 0, diff_height, 0, diff_depth])
        return x


# # Testing the model
# if __name__ == "__main__":
#     # Sample input: batch_size=1, channels=3, depth=16, height=64, width=64
#     input_tensor = torch.randn(1, 3, 16, 64, 64)

#     model = Network()
#     output = model(input_tensor)

#     print("Final Output Shape:", output.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Network(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_channels=32):
#         super(Network, self).__init__()

#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm3d(base_channels),
#             nn.ReLU()
#         )

#         self.encoder2 = nn.Sequential(
#             nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 2),
#             nn.ReLU()
#         )

#         self.encoder3 = nn.Sequential(
#             nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 4),
#             nn.ReLU()
#         )

#         self.encoder4 = nn.Sequential(
#             nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 8),
#             nn.ReLU()
#         )

#         self.encoder5 = nn.Sequential(
#             nn.Conv3d(base_channels * 8, base_channels * 16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(base_channels * 16),
#             nn.ReLU()
#         )

#         # Decoder
#         self.decoder5 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels * 8),
#             nn.ReLU()
#         )

#         self.decoder4 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels * 4),
#             nn.ReLU()
#         )

#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels * 2),
#             nn.ReLU()
#         )

#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm3d(base_channels),
#             nn.ReLU()
#         )

#         self.decoder1 = nn.Sequential(
#             nn.Conv3d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()  # To scale output to [0, 1]
#         )

#     def forward(self, x):
#         input_shape = x.shape  # Store input shape for padding at the end

#         # Encoder pass
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#         x4 = self.encoder4(x3)
#         x5 = self.encoder5(x4)

#         # Decoder pass
#         d5 = self.decoder5(x5)
#         d4 = self.decoder4(d5)
#         d3 = self.decoder3(d4)
#         d2 = self.decoder2(d3)
#         out = self.decoder1(d2)

#         # Pad output if necessary to match input size
#         diff_depth = input_shape[2] - out.shape[2]
#         diff_height = input_shape[3] - out.shape[3]
#         diff_width = input_shape[4] - out.shape[4]

#         if diff_depth != 0 or diff_height != 0 or diff_width != 0:
#             out = F.pad(out, [0, diff_width, 0, diff_height, 0, diff_depth])

#         return out

