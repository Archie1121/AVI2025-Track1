import torch
import torch.nn as nn


class HybridChunkProjector(nn.Module):
    def __init__(self, input_dim, samples, output_dim, dropout, use_prompt=True):
        super().__init__()

        self.input_dim = input_dim*samples
        self.output_dim = output_dim
        self.chunks_num = self._auto_adjust_chunks(self.input_dim)
        assert self.input_dim % self.chunks_num == 0
        chunk_size_in = self.input_dim // self.chunks_num   # 512
        # print(self.input_dim)
        # print(f"Chunk size in: {chunk_size_in}")
        chunk_size_out = self.output_dim // self.chunks_num
        # print(f"Chunk size out: {chunk_size_out}")
        self.use_prompt = use_prompt
        self.dropout = dropout
        

        self.chunk_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(chunk_size_in, chunk_size_out*4),    # 512 -> 1024
                nn.ReLU(),
                nn.LayerNorm(chunk_size_out*4),
                nn.Linear(chunk_size_out*4, chunk_size_out),   # 1024 -> 256
                nn.Dropout(self.dropout)
            ) for _ in range(self.chunks_num)
        ])
        

        self.use_prompt = self.use_prompt
        if self.use_prompt:
            self.prompt_tokens = nn.ParameterList([
                nn.Parameter(torch.randn(1, chunk_size_out)) 
                for _ in range(self.chunks_num)
            ])
            self.prompt_gate = nn.Sequential(
                nn.Linear(chunk_size_out*2, 1),
                nn.Sigmoid()
            )
        

        self.global_compensate = nn.Parameter(torch.randn(output_dim))
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.chunk_layers:
            nn.init.kaiming_normal_(
                layer[0].weight, 
                mode='fan_in', 
                nonlinearity='relu'
            )
            nn.init.zeros_(layer[0].bias)
            
            nn.init.xavier_normal_(
                layer[3].weight, 
                gain=nn.init.calculate_gain('linear')*0.5
            )
            nn.init.normal_(layer[3].bias, std=0.01)

        if self.use_prompt:
            for token in self.prompt_tokens:
                nn.init.normal_(token, mean=0.0, std=0.02)

        nn.init.orthogonal_(self.prompt_gate[0].weight)
        nn.init.constant_(self.prompt_gate[0].bias, 0.5)

        nn.init.constant_(self.global_compensate, 0.1) 

    def _auto_adjust_chunks(self,input_dim):
        if self.input_dim >= 20480:
            return 40
        if self.input_dim >= 16384:
            return 32
        elif self.input_dim >= 8192:
            return 16
        elif self.input_dim >= 4096:
            return 8
        elif self.input_dim >= 2048:
            return 4
        elif self.input_dim >= 1024:
            return 2
        else:
            return 1

    def forward(self, x):
        
        x = x.squeeze(1)
        chunks = torch.chunk(x, self.chunks_num, dim=1)
        # print(f"Chunks: {len(chunks)}")
        # print(f"Chunk size: {chunks[0].size()}")
        processed = []
        
        for i, (chunk, layer) in enumerate(zip(chunks, self.chunk_layers)):
            feat = layer(chunk)
            # print(f"Chunk {i} processed size: {feat.size()}") 
            
            if self.use_prompt:
                prompt = self.prompt_tokens[i].expand(feat.size(0), -1)
                gate = self.prompt_gate(torch.cat([feat, prompt], dim=1))
                feat = gate * feat + (1 - gate) * prompt
                
            processed.append(feat)
        
        output = torch.cat(processed, dim=1) + self.global_compensate
        # print(f"Output size: {output.size()}")
        return output