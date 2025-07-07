import torch.nn as nn
import torch
import torch.nn.functional as F
from .ATconnector import ATconnector
from .VTconnector import VTconnector
from .TextFeatureEnhancer import TextFeatureEnhancer
from .Token_Refinement_Module import HybridChunkProjector


class FusionModel(nn.Module):
    
    '''
    args:
        audio_dim: audio feature dimension
        video_dim: video feature dimension
        text_dim: text feature dimension
        unified_dim: unified feature dimension
        hidden_dim: hidden feature dimension
        enhancer_dim: enhancer feature dimension
        target_dim: target prediction dimension
    '''
    def __init__(self, args):
        super(FusionModel, self).__init__()
        self.args = args
        
        self.audio_projector = HybridChunkProjector(
            input_dim=args.audio_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_audio,
            use_prompt=args.use_prompt
        )
        
        self.video_projector = HybridChunkProjector(
            input_dim=args.video_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_video,
            use_prompt=args.use_prompt
        )
        
        self.text_projector1 = HybridChunkProjector(
            input_dim=args.text_dim,
            samples=1,
            output_dim=args.unified_dim,
            dropout=args.HCPdropout_text,
            use_prompt=args.use_prompt
        )

        self.text_projector2 = HybridChunkProjector(
            input_dim=args.text_dim,
            samples=1,
            output_dim=args.hidden_dim,
            dropout=args.HCPdropout_pure_text,
            use_prompt=args.use_prompt
        )

        self.at_connector = ATconnector(
            n_heads= args.heads_num,
            dim_audio=args.unified_dim,
            dim_text=args.unified_dim,
            output_dim=args.hidden_dim,
            dropout = args.ATCdropout
        )
        
        self.vt_connector = VTconnector(
            n_heads= args.heads_num,
            dim_video=args.unified_dim,
            dim_text=args.unified_dim,
            output_dim=args.hidden_dim,
            dropout = args.VTCdropout
        )

        self.text_enhancer = TextFeatureEnhancer(
            feat_dim=args.hidden_dim,
            out_dim=args.enhancer_dim,
            hidden_dim=args.enhancer_dim,
            dropout= args.TFEdropout
        )

        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.enhancer_dim, args.enhancer_dim // 2),
                nn.ReLU(),
                nn.Linear(args.enhancer_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, args.target_dim)
            ) for _ in range(32)
        ])

    def forward(self, audio_feat, video_feat, text_feat):

        # Step 1: feature alignment    
        audio_proj = self.audio_projector(audio_feat)
        video_proj = self.video_projector(video_feat) 
        text_proj1 = self.text_projector1(text_feat)

        text_proj2 = self.text_projector2(text_feat)

        # Step 2: cross-modal interaction
        at_fusion = self.at_connector(audio_proj, text_proj1)
        vt_fusion = self.vt_connector(video_proj, text_proj1)

        # Step 3: text feature enhancement
        enhanced_text = self.text_enhancer(
            t_feat = text_proj2,
            at_feat=at_fusion,
            vt_feat=vt_fusion
        )
        # print(enhanced_text.shape)

        # Step 4: ensemble prediction
        predictions = torch.stack([mlp(enhanced_text) for mlp in self.ensemble], dim=0)
        prediction = predictions.mean(dim=0)  # shape: [batch_size, target_dim]
        
        return prediction

    def compute_loss(self, pred, target):
        return nn.MSELoss()(pred, target)