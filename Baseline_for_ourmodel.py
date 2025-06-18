import torch
import torch.nn as nn
import torchaudio

# ----- Audio Encoder (HuBERT-style Stub) -----
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

# ----- Video Encoder (simple Conv + Transformer stub) -----
class VideoEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, C, T, 1, 1)
        return x.squeeze(-1).squeeze(-1)  # (B, C, T)

# ----- Cross Attention -----
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attn(query, key, value)
        return out

# ----- Fusion + Temporal Model -----
class TemporalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True),
            num_layers=2
        )

    def forward(self, x):
        return self.temporal(x)

# ----- Audio Decoder (Basic Conv Decoder) -----
class AudioDecoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=80):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

# ----- Full Model -----
class AVDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.video_encoder = VideoEncoder()
        self.cross_attn_1 = CrossAttention(dim=256)
        self.cross_attn_2 = CrossAttention(dim=256)
        self.temporal_fusion = TemporalFusion(dim=256)
        self.audio_decoder = AudioDecoder()

    def forward(self, audioL, audioR, video):
        zL = self.audio_encoder(audioL)  # (B, C, T)
        zR = self.audio_encoder(audioR)
        zV = self.video_encoder(video)  # (B, C, T)

        zL = zL.transpose(1, 2)  # (B, T, C)
        zR = zR.transpose(1, 2)
        zV = zV.transpose(1, 2)

        zA = self.cross_attn_1(zL, zR, zR)
        zV_attended = self.cross_attn_2(zR, zV, zV)

        fused = torch.cat([zA, zV_attended], dim=-1)  # (B, T, 2C)
        fused = self.temporal_fusion(fused)
        fused = fused.transpose(1, 2)  # (B, C, T)

        clean_audio = self.audio_decoder(fused)  # (B, 80, T)
        return clean_audio

# ----- Example Usage -----
if __name__ == '__main__':
    model = AVDenoiser()
    B, C, T = 2, 80, 16000
    video = torch.randn(B, 3, 16, 56, 56)  # B x C x T x H x W
    audioL = torch.randn(B, C, T)
    audioR = torch.randn(B, C, T)
    out = model(audioL, audioR, video)
    print(out.shape)  # Expect: (B, 80, T)
