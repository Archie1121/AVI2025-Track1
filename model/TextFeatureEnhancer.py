import torch
import torch.nn as nn

class TextFeatureEnhancer(nn.Module):
    def __init__(self, feat_dim=1024, hidden_dim=512, out_dim=1024, dropout=0.2):
        super().__init__()

        self.feat_dim = feat_dim
        self.out_dim = out_dim

        # 门控权重生成器
        self.gate_controller = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 为 AT/VT/T 生成3个权重
            nn.Softmax(dim=-1)
        )

        self.at_proj = nn.Linear(feat_dim, feat_dim)
        self.vt_proj = nn.Linear(feat_dim, feat_dim)
        self.t_proj = nn.Linear(feat_dim, feat_dim)

        self.out_norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

        # 控制输出维度（如果不同于输入维度）
        if out_dim != feat_dim:
            self.output_proj = nn.Linear(feat_dim, out_dim)
        else:
            self.output_proj = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for proj in [self.at_proj, self.vt_proj, self.t_proj]:
            nn.init.kaiming_normal_(proj.weight, mode='fan_out')
            nn.init.zeros_(proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.kaiming_normal_(self.output_proj.weight, mode='fan_out')
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, at_feat, vt_feat, t_feat):
        """
        输入:
            at_feat: [B, feat_dim]  (Audio-Text融合特征)
            vt_feat: [B, feat_dim]  (Video-Text融合特征)
            t_feat:  [B, feat_dim]  (原始Text特征)
        输出:
            [B, out_dim] 增强后的文本特征
        """
        # 1. 动态门控权重生成
        concat_feats = torch.cat([at_feat, vt_feat, t_feat], dim=-1)  # [B, feat_dim * 3]
        gates = self.gate_controller(concat_feats)  # [B, 3]

        # 2. 特征变换
        at_trans = self.at_proj(at_feat)  # [B, feat_dim]
        vt_trans = self.vt_proj(vt_feat)
        t_trans = self.t_proj(t_feat)

        # 3. 门控加权融合
        weighted_feats = (
            gates[:, 0].unsqueeze(-1) * at_trans +
            gates[:, 1].unsqueeze(-1) * vt_trans +
            gates[:, 2].unsqueeze(-1) * t_trans
        )

        # 4. 残差连接 + 正规化 + 映射到目标输出维度
        enhanced = self.out_norm(t_feat + self.dropout(weighted_feats))  # [B, feat_dim]
        return self.output_proj(enhanced)  # [B, out_dim]
