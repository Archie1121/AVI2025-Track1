import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class VTconnector(nn.Module):
    def __init__(self, n_heads, dim_video, dim_text, output_dim, dropout=0.1):
        """
        参数说明：
            n_heads: 注意力头数
            dim_video: 视频输入维度如1024
            dim_text: 文本输入维度如768
            output_dim: 输出维度需能被n_heads整除
        """
        super().__init__()
        assert output_dim % n_heads == 0, "output_dim must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_head = output_dim // n_heads
        
        # 投影层（Video作为Query，Text作为Key/Value）
        self.q_proj = nn.Linear(dim_video, output_dim)  # 视频作为查询
        self.k_proj = nn.Linear(dim_text, output_dim)   # 文本作为键
        self.v_proj = nn.Linear(dim_text, output_dim)  # 文本作为值
        
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

    def forward(self, x_video, x_text):
        """
        输入:
            x_video: [batch_size, dim_video] (如 [32, 1024])
            x_text:  [batch_size, dim_text]  (如 [32, 768])
        输出:
            [batch_size, output_dim]
        """
        B = x_video.size(0)
        
        # 1. 投影到多头空间 [B, H, D/H]
        q = self.q_proj(x_video).view(B, self.n_heads, self.d_head)  # 视频作为Query
        k = self.k_proj(x_text).view(B, self.n_heads, self.d_head)   # 文本作为Key
        v = self.v_proj(x_text).view(B, self.n_heads, self.d_head)    # 文本作为Value
        
        # 2. 计算注意力分数 [B, H]
        scores = (q * k).sum(dim=-1) / (self.d_head ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # 3. 加权Value [B, H, D/H]
        context = attn.unsqueeze(-1) * v
        
        # 4. 拼接所有head [B, output_dim]
        context = context.reshape(B, -1)
        
        return self.out(context)