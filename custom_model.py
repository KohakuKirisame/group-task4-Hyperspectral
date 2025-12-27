import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, dim))

    def forward(self, x):
        return x + self.pos


class ConvTransformerAE(nn.Module):
    def __init__(self,
        input_dim=32,
        latent_dim=4,
        conv_channels=4,
        d_model=32,
        n_head=4,
        num_layers=2,
        dropout=0.1,
        kernel_size=3):

        super().__init__()
        self.input_dim = input_dim

        # 卷积
        padding = (kernel_size - 1) // 2 # padding = 1
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.GELU()
        )

        self.embedding = nn.Linear(conv_channels, d_model)
        self.position_embedding = PositionalEmbedding(32, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(input_dim, input_dim // 2),
        #     nn.GELU(),
        #     nn.Linear(input_dim // 2, latent_dim)
        # )


        self.to_latent = nn.Linear(d_model, latent_dim)

        self.from_latent = nn.Linear(latent_dim, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, input_dim)
        )

        self.reconstruction = nn.Linear(d_model, 1)

    def encode(self, x):
        # (B, 32)
        x_raw = x
        x = x.unsqueeze(1) # (B, 1, 32)
        x = self.conv(x) # (B, C, 32)
        x = x.transpose(1, 2) # (B, 32, C)
        x = self.embedding(x) # (B, 32, d_model)
        x = self.position_embedding(x)

        x = self.encoder(x)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean = torch.squeeze(x_mean, dim=1)
        x = self.to_latent(x_mean) # (B, 4)
        # x = self.encoder(x)
        return x

    def decode(self, x):
        # x = self.from_latent(x)
        # x = x.unsqueeze(1).repeat(1, self.input_dim, 1)

        x = self.decoder(x)
        # x = self.reconstruction(x)
        return x
        # return x.squeeze(-1) # (B, 32)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x, x_hat, z


class CustomClassifier(nn.Module):
    def __init__(self,
        latent_dim=16,
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class NewCustomClassifier(nn.Module):
    def __init__(self,
        latent_dim=16,
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class AEClassifier(nn.Module):
    def __init__(self, ae_model, classifier_model):
        super().__init__()
        self.ae_model = ae_model
        self.classifier_model = classifier_model

    def forward(self, x):
        # with torch.no_grad():
        x = self.ae_model.encode(x)
        x_hat = self.ae_model.decode(x)

        logits = self.classifier_model(x)
        return logits, x_hat

    def fit(self, x):
        x = self.ae_model.encode(x)
        x = self.classifier_model(x)
        return x