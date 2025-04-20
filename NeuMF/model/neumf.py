import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TemporalGMF(nn.Module):
    def __init__(self, embedding_dim, temporal_dim):
        super().__init__()
        self.temporal_transform = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.gmf_layer = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_emb, item_emb, temporal_emb):
        # Transform temporal embedding
        temporal_transformed = self.temporal_transform(temporal_emb)
        
        # Element-wise multiplication with temporal information
        gmf_output = user_emb * item_emb * temporal_transformed
        return self.gmf_layer(gmf_output)

class TimeAwareAttention(nn.Module):
    def __init__(self, dim, temporal_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Time-aware projection
        self.time_proj = nn.Sequential(
            nn.Linear(temporal_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
        # Time-aware attention components
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Time-based position encoding
        self.time_pe = nn.Parameter(torch.randn(1, 1, dim))
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, context, time_embedding):
        B = x.shape[0]  # batch size
        residual = x
        
        # Add time position encoding
        x = x + self.time_pe
        
        # Project time embeddings
        time_context = self.time_proj(time_embedding)
        
        # Self-attention with time context
        q = self.q(self.norm1(x)).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context + time_context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context + time_context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with time awareness
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Add residual and normalize
        x = x + residual
        x = self.norm2(x)
        return x

class TemporalFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Temporal gating
        self.time_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Feed-forward network with gating
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
        self.gate_ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_emb):
        B = x.shape[0]
        
        # Ensure time_emb has the same dimension as x
        if time_emb.dim() == 3 and time_emb.size(-1) != self.dim:
            time_emb = self.time_gate[0](time_emb)  # Project to correct dimension
        
        # Self-attention with time context
        q = self.q(x).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(time_emb).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(time_emb).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Temporal gating with proper reshaping
        time_context = time_emb.mean(dim=1, keepdim=True)  # [B, 1, D]
        gate = self.time_gate(time_context)
        x = gate * x
        
        # First residual
        x = x + self.norm1(x)
        
        # Feed-forward network with gating
        residual = x
        x = self.ffn(x)
        gate_ffn = self.gate_ffn(residual)
        x = gate_ffn * x
        x = self.dropout(x)
        
        # Second residual
        x = x + self.norm2(residual)
        
        return x

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128, num_heads=16, dropout=0.3):
        super(NeuMF, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        
        # Enhanced embeddings with larger dimensions
        self.user_embedding = nn.Embedding(num_users, embedding_dim * 2)
        self.item_embedding = nn.Embedding(num_items, embedding_dim * 2)
        
        # Improved normalization with layer norm
        self.user_norm = nn.LayerNorm(embedding_dim * 2)
        self.item_norm = nn.LayerNorm(embedding_dim * 2)
        
        # Enhanced time encoding
        self.time_encoding = nn.Sequential(
            nn.Linear(128, embedding_dim * 4),
            nn.LayerNorm(embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2)
        )
        
        # Multi-scale temporal fusion
        self.temporal_fusion_layers = nn.ModuleList([
            TemporalFusion(embedding_dim * 2, num_heads, dropout)
            for _ in range(3)  # Multiple layers for hierarchical processing
        ])
        
        # Enhanced cross attention
        self.cross_attention_layers = nn.ModuleList([
            TemporalFusion(embedding_dim * 2, num_heads, dropout)
            for _ in range(3)  # Multiple layers
        ])
        
        # Improved TGN feature projection
        self.tgn_projection = nn.Sequential(
            nn.Linear(128, embedding_dim * 4),
            nn.LayerNorm(embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2)
        )
        
        # Enhanced feature fusion with multi-head attention
        self.fusion_attention = MultiHeadTemporalAttention(embedding_dim * 2, num_heads)
        
        # Residual fusion gates
        self.user_fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.Sigmoid()
        )
        
        self.item_fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.Sigmoid()
        )
        
        # Enhanced MLP with residual connections and layer normalization
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 4, embedding_dim * 4),
                nn.LayerNorm(embedding_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(embedding_dim * 4, embedding_dim * 2),
                nn.LayerNorm(embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])
        
        # Final prediction layers with attention
        self.attention_pooling = MultiHeadTemporalAttention(embedding_dim, num_heads)
        self.final_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def predict_from_embeddings(self, user_emb, item_emb, temporal_emb):
        """Enhanced prediction with multi-scale temporal fusion and cross attention."""
        batch_size = user_emb.shape[0]
        
        # Normalize embeddings
        user_emb = self.user_norm(user_emb)
        item_emb = self.item_norm(item_emb)
        
        # Enhanced temporal processing
        time_emb = self.time_encoding(temporal_emb)
        time_emb = time_emb.unsqueeze(1)
        
        # Multi-scale temporal fusion
        user_temporal = user_emb.unsqueeze(1)
        item_temporal = item_emb.unsqueeze(1)
        
        for temporal_layer in self.temporal_fusion_layers:
            user_temporal = temporal_layer(user_temporal, time_emb)
            item_temporal = temporal_layer(item_temporal, time_emb)
        
        # Enhanced cross attention
        for cross_layer in self.cross_attention_layers:
            user_temporal = cross_layer(user_temporal, item_temporal)
            item_temporal = cross_layer(item_temporal, user_temporal)
        
        # Project TGN features
        tgn_features = self.tgn_projection(temporal_emb)
        
        # Enhanced feature fusion with attention
        user_temporal = user_temporal.squeeze(1)
        item_temporal = item_temporal.squeeze(1)
        
        # Compute fusion gates with enhanced context
        user_gate = self.user_fusion_gate(torch.cat([user_temporal, tgn_features], dim=-1))
        item_gate = self.item_fusion_gate(torch.cat([item_temporal, tgn_features], dim=-1))
        
        # Apply gated fusion
        user_final = user_temporal + user_gate * tgn_features
        item_final = item_temporal + item_gate * tgn_features
        
        # Combine features with attention
        combined = torch.cat([user_final, item_final], dim=-1)
        
        # Enhanced MLP processing with residual connections
        x = combined
        for layer in self.mlp:
            residual = x if x.shape == layer(x).shape else None
            x = layer(x)
            if residual is not None:
                x = x + residual
        
        # Final attention pooling and prediction
        x = x.unsqueeze(1)
        x = self.attention_pooling(x)
        x = x.squeeze(1)
        
        # Final projection with sigmoid activation
        prediction = self.final_projection(x)
        return torch.sigmoid(prediction).squeeze(-1)
    
    def forward(self, user, item, temporal_emb):
        # Get base embeddings
        user_emb = self.user_embedding(user)  # [B, D]
        item_emb = self.item_embedding(item)  # [B, D]
        
        return self.predict_from_embeddings(user_emb, item_emb, temporal_emb) 