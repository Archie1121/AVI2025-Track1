import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class ATconnector(nn.Module):
    def __init__(self, n_heads, dim_text, dim_audio, output_dim, dropout=0.1):
        super().__init__()
        assert output_dim % n_heads == 0, "output_dim must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_head = output_dim // n_heads
        
        # 投影层（无序列维度）
        self.q_proj = nn.Linear(dim_text, output_dim)   # 10240 -> 5120
        self.k_proj = nn.Linear(dim_audio, output_dim)
        self.v_proj = nn.Linear(dim_audio, output_dim)
        
        # 输出处理
        self.out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, x_text, x_video):
        """
        输入:
            x_text:  [batch_size, dim_text] (如 [32, 768])
            x_video: [batch_size, dim_video] (如 [32, 1024])
        输出:
            [batch_size, output_dim]
        """
        B = x_text.size(0)
        
        # 1. 投影到多头空间 [B, H, D/H]
        q = self.q_proj(x_text).view(B, self.n_heads, self.d_head)
        k = self.k_proj(x_video).view(B, self.n_heads, self.d_head)
        v = self.v_proj(x_video).view(B, self.n_heads, self.d_head)
        
        # 2. 计算注意力分数 [B, H]
        scores = (q * k).sum(dim=-1) / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)  # 在head维度归一化
        
        # 3. 加权value [B, H, D/H]
        context = attn.unsqueeze(-1) * v
        
        # 4. 拼接所有head [B, output_dim]
        context = context.reshape(B, -1)
        
        return self.out(context)