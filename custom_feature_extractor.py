from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th, torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

class FiLMExtractor1(nn.Module):
    def __init__(self, sta_feat_dim, context_feat_dim, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.sta_feat_dim = sta_feat_dim
        self.context_feat_dim = context_feat_dim
        self.embed_dim = embed_dim

        # STA feature pre-embedding
        self.sta_embedding = nn.Linear(sta_feat_dim, embed_dim)

        # FiLM conditioning parameters from context
        self.film_gamma = nn.Linear(context_feat_dim, embed_dim)
        self.film_beta = nn.Linear(context_feat_dim, embed_dim)

        # Attention pooling
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

        # Final projection (you can expand this if needed)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # x["sta"]: shape (B, S, F) — STA features
        # x["context"]: shape (B, C) — context vector
        sta = x["sta"]         # (B, S, F)
        context = x["context"] # (B, C)

        # Step 1: Project STA features
        sta_embed = self.sta_embedding(sta)  # (B, S, D)

        # Step 2: FiLM conditioning
        gamma = self.film_gamma(context).unsqueeze(1)  # (B, 1, D)
        beta = self.film_beta(context).unsqueeze(1)    # (B, 1, D)

        modulated = sta_embed * gamma + beta  # (B, S, D)

        # Step 3: Apply attention across STAs
        attn_output, _ = self.attn(modulated, modulated, modulated)  # (B, S, D)

        # Step 4: Pool across STAs (mean pooling)
        pooled = attn_output.mean(dim=1)  # (B, D)

        # Step 5: Final output projection
        return self.output_layer(pooled)
    


class FiLMExtractor2(BaseFeaturesExtractor):     # TOP
    def __init__(self, obs_space, dyn_dim, stat_dim):
        super().__init__(obs_space, features_dim=dyn_dim)
        self.ln = nn.LayerNorm(dyn_dim)
        self.embed_s = nn.Sequential(nn.Linear(stat_dim, 32), nn.ReLU())
        self.to_gamma = nn.Linear(32, dyn_dim)
        self.to_beta  = nn.Linear(32, dyn_dim)
        # warm-start
        nn.init.constant_(self.to_gamma.bias, 1.0)
        nn.init.constant_(self.to_beta.bias,  0.0)

    def forward(self, obs):
        x_dyn  = obs["dynamic"]
        x_stat = obs["static"]
        e = self.embed_s(x_stat)
        gamma = self.to_gamma(e)
        beta  = self.to_beta(e)
        x = self.ln(x_dyn)
        # residual FiLM
        return x_dyn + (gamma * x + beta)
    
class FiLMExtractor3(BaseFeaturesExtractor):             # version of FiLMExtractor2
    def __init__(self, obs_space, dyn_dim, stat_dim):
        super().__init__(obs_space, features_dim=dyn_dim)
        self.ln_dyn  = nn.LayerNorm(dyn_dim)
        self.ln_stat = nn.LayerNorm(stat_dim)
        self.embed_s = nn.Sequential(
            nn.Linear(stat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.to_gamma = nn.Linear(32, dyn_dim)
        self.to_beta  = nn.Linear(32, dyn_dim)
        self.film_alpha = nn.Parameter(th.zeros(1))  # scalar gate
        # warm start
        nn.init.constant_(self.to_gamma.bias, 1.0)
        nn.init.constant_(self.to_beta.bias,  0.0)

    def forward(self, obs):
        x_dyn  = obs["dynamic"]
        x_stat = self.ln_stat(obs["static"])
        e = self.embed_s(x_stat)
        gamma = self.to_gamma(e)
        beta  = self.to_beta(e)
        x = self.ln_dyn(x_dyn)
        modulated = gamma * x + beta
        return x_dyn + th.sigmoid(self.film_alpha) * modulated
    
class FiLMExtractor4(BaseFeaturesExtractor):
    def __init__(self, obs_space, dyn_dim, stat_dim):
        super().__init__(obs_space, features_dim=dyn_dim)
        self.ln = nn.LayerNorm(dyn_dim)

        # keep it shallow
        self.embed_s = nn.Sequential(
            nn.Linear(stat_dim, 32),
            nn.ReLU()
        )

        self.to_gamma = nn.Linear(32, dyn_dim)
        self.to_beta  = nn.Linear(32, dyn_dim)

        # warm start
        nn.init.constant_(self.to_gamma.bias, 1.0)
        nn.init.constant_(self.to_beta.bias,  0.0)

    def forward(self, obs):
        x_dyn  = obs["dynamic"]
        x_stat = obs["static"]
        e = self.embed_s(x_stat)
        gamma = self.to_gamma(e)
        beta  = self.to_beta(e)
        x = self.ln(x_dyn)
        return x_dyn + (gamma * x + beta)
    
class ProgressiveFiLMExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, dyn_dim, stat_dim):
        super().__init__(obs_space, features_dim=dyn_dim)
        # FiLM submodules (as before)…
        self.gamma_net = nn.Linear(stat_dim, dyn_dim)
        self.beta_net  = nn.Linear(stat_dim, dyn_dim)
        # Gate parameter (learnable)
        self.alpha = nn.Parameter(th.zeros(1))  # starts at 0
        # init gamma bias =1, beta bias =0
        nn.init.constant_(self.gamma_net.bias, 1.0)
        nn.init.constant_(self.beta_net.bias,  0.0)

    def forward(self, obs):
        x_dyn  = obs["dynamic"]
        x_stat = obs["static"]
        gamma = self.gamma_net(x_stat)
        beta  = self.beta_net(x_stat)
        film  = gamma * x_dyn + beta
        # progressive blend
        return x_dyn * (1 - th.sigmoid(self.alpha)) + film * th.sigmoid(self.alpha)
    
class SharedMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box,
                 num_stas: int = 16, num_aps: int = 4,  # adjust default as needed
                 hidden_dim: int = 64,
                 pooling: str = "mean",
                 **kwargs):
        total_dim = observation_space.shape[0]
        expected_dim = 2 * num_stas + num_stas * num_aps
        assert total_dim == expected_dim, f"Observation shape mismatch: got {total_dim}, expected {expected_dim}"
        
        self.num_stas = num_stas
        self.num_aps = num_aps
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        super().__init__(observation_space, features_dim=hidden_dim)

        # MLP for each STA's dynamic features (delay + queue)
        self.dynamic_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # MLP for each STA's per-AP channel coefficients
        self.channel_mlp = nn.Sequential(
            nn.Linear(num_aps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Combine both
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.shape[0]
        N = self.num_stas
        M = self.num_aps

        # Slice inputs
        delays = obs[:, :N]                   # [B, N]
        queues = obs[:, N:2*N]                # [B, N]
        channels = obs[:, 2*N:]               # [B, N*M]

        # Combine dynamic per-STA info → [B, N, 2]
        dynamic_features = th.stack([delays, queues], dim=2)  # [B, N, 2]
        dyn_out = self.dynamic_mlp(dynamic_features)          # [B, N, hidden_dim]

        # Reshape channels into [B, N, M] and apply per-STA channel MLP
        channel_matrix = channels.view(B, N, M)               # [B, N, M]
        chan_out = self.channel_mlp(channel_matrix)           # [B, N, hidden_dim]

        # Concatenate per-STA features: [B, N, 2*hidden_dim]
        per_sta_feat = th.cat([dyn_out, chan_out], dim=2)

        # Combine and pool
        fused = self.final_mlp(per_sta_feat)                  # [B, N, hidden_dim]
        
        if self.pooling == "mean":
            pooled = fused.mean(dim=1)                        # [B, hidden_dim]
        elif self.pooling == "max":
            pooled = fused.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")

        return pooled
    
class SharedMLPWithAttentionExtractor(nn.Module):
    def __init__(self, observation_space: spaces.Box, sta_number: int, ap_number: int, hidden_dim: int = 64):
        super().__init__()
        self.sta_number = sta_number
        self.ap_number = ap_number
        self.per_sta_dim = 2 + ap_number  # delay, queue, and AP channel vector
        
        # Add LayerNorm for input stabilization
        self.layernorm = nn.LayerNorm(self.per_sta_dim)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.per_sta_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm between MLP layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.attention_net = nn.Linear(hidden_dim, 1)
        self._features_dim = hidden_dim  # Output is [B, hidden_dim]

    def forward(self, obs: th.Tensor) -> th.Tensor:
        B = obs.size(0)
        N_STA = self.sta_number
        N_AP = self.ap_number
        
        # Efficient reshaping: [B, N_STA, (2 + N_AP)]
        x = obs.view(B, N_STA, -1)
        
        # Apply LayerNorm per STA feature
        x = self.layernorm(x)
        
        sta_features = self.shared_mlp(x)  # [B, N_STA, hidden_dim]
        
        # Attention with temperature scaling
        scores = self.attention_net(sta_features) / (self._features_dim ** 0.5)
        weights = th.softmax(scores, dim=1)  # [B, N_STA, 1]
        
        pooled = th.sum(sta_features * weights, dim=1)  # [B, hidden_dim]
        return pooled

    # ... (keep other methods unchanged)
    def forward_shared(self, obs: th.Tensor) -> th.Tensor:
        return self.forward(obs)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features

    @property
    def features_dim(self) -> int:
        return self._features_dim