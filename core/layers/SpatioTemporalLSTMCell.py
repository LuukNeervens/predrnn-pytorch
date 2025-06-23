__author__ = 'yunbo'

import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=8):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = num_hidden // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Conv2d(num_hidden, num_hidden, 1, bias=False)
        self.key = nn.Conv2d(num_hidden, num_hidden, 1, bias=False)
        self.value = nn.Conv2d(num_hidden, num_hidden, 1, bias=False)
        self.out_proj = nn.Conv2d(num_hidden, num_hidden, 1, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention computation
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) * self.scale, dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        
        # Reshape and project
        out = out.view(B, C, H, W)
        return self.out_proj(out)

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm, use_attention=False, attention_heads=8):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.use_attention = use_attention
        
        # Original convolution layers
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Add attention mechanism conditionally
        if self.use_attention:
            self.attention = SpatialAttention(num_hidden, attention_heads)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_concat = self.conv_c(c_t)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c, f_c, g_c = torch.split(c_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_c)
        f_t_prime = torch.sigmoid(f_x_prime + f_c + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_c)

        c_t_prime = f_t_prime * c_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, c_t_prime), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        # Apply attention to hidden state if enabled
        if self.use_attention:
            h_attended = self.attention(h_new)
            h_new = h_new + h_attended  # Residual connection

        return h_new, c_new









