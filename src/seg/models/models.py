import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, Up, OutConv

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000, dropout_prob=0.1):
        r"""
        Inject some information about the absolute or relative information about each
        element in the sequence using the sine and cosine functions with different frequencies.
        This is exactly as in the original paper on Attention Is All You Need.
        ..math:
          \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
          \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
          \text{where pos is the word position and i is the embed idx)
        Args:
          d_model: The embedding dimension (required).
          max_len: The maximum length of the input sequence (default=5000)
          dropout_prob: The probability of dropping any neuron (default=0.1)
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # assign the even coordinates
        pe[0, :, 1::2] = torch.cos(position * div_term)  # odd coordinates

        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward funcion
        Args:
          x: the sequence fed to the positional encoder model (required).
        Shape:
          x: [N, T, d_model]
          output: [N, T, d_model]
        Examples:
          >>> output = pos_encoder(x)
        """
        # x: N, T, d_model
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class UNetEncoder(nn.Module):
    def __repr__(self):
        return (
            f"UNetEncoder-m{self.d_model}-h{self.n_heads}-l{self.n_layers}"
            f"-k{self.kernel_size}-c{self.n_classes}-f{self.input_dim}"
        )

    def __init__(
        self,
        max_len,
        d_model,
        n_heads,
        n_layers,
        n_classes,
        n_features,
        dropout_prob,
        kernel_size=3,
        feedforward_mult=4,
    ):
        if d_model % 16 != 0:
            raise ValueError(f"{d_model=} must be a multiple of 16")
        super().__init__()
        self.input_dim = n_features
        self.max_len = max_len
        self.d_model = d_model
        self.downsample = 16
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.kernel_size = kernel_size

        self.inc = DoubleConv(self.input_dim, d_model // 16, kernel_size=kernel_size)
        self.down1 = Down(d_model // 16, d_model // 8, kernel_size=kernel_size)
        self.down2 = Down(d_model // 8, d_model // 4, kernel_size=kernel_size)
        self.down3 = Down(d_model // 4, d_model // 2, kernel_size=kernel_size)
        self.down4 = Down(d_model // 2, d_model, kernel_size=kernel_size)

        self.pos_encoding = PositionalEncoding(
            self.d_model, max_len // self.downsample, dropout_prob
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=int(feedforward_mult * self.d_model),
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=None,
        )
        self.ln = torch.nn.LayerNorm(self.d_model)

        self.up1 = Up(d_model, d_model // 2, kernel_size=kernel_size)
        self.up2 = Up(d_model // 2, d_model // 4, kernel_size=kernel_size)
        self.up3 = Up(d_model // 4, d_model // 8, kernel_size=kernel_size)
        self.up4 = Up(d_model // 8, d_model // 16, kernel_size=kernel_size)
        self.outc = OutConv(d_model // 16, n_classes)

    def forward(self, x):
        """
        x: N, F, T -> N, T, C

        where
          N - Batch size.
          F - Number of input features.
          T - Sequence length.
          C - Number of output classes.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)  # N, d_model, T/s

        x = x.permute(0, 2, 1)  # N, T/s, d_model
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.ln(x)

        x = x.permute(0, 2, 1)  # N, d_model, T/s
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)  # N, d_model/s, T
        logits = self.outc(x)  # N, C, T
        logits = logits.permute(0, 2, 1)
        return logits
