from torch import nn
import torch
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
from ur_lstm import UR_LSTM
try:
    import mamba_ssm
except:
    print("Failed importing mamba_ssm.")

def sequence_length(subsample, n_test, repeat_test):
    return 90*300//subsample+2*n_test*repeat_test

class Model(nn.Module):
    def __init__(self, model, dim: int = 512, d_embed: int = 512, n_test: int = 90, repeat_test: int = 1, **kwargs):
        super().__init__()
        self.embed = nn.Linear(dim, d_embed)
        self.model = model(d_embed = d_embed, n_test = n_test, repeat_test = repeat_test, **kwargs)
        self.output = nn.Linear(d_embed, 1)
        self.sigmoid = nn.Sigmoid()
        self.n_test = n_test
        self.repeat_test = repeat_test

    def forward(self, x):
        x = self.embed(x)
        x = self.model(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)[..., -2*self.n_test*self.repeat_test+self.repeat_test-1::self.repeat_test]

class TransformerEncoder(nn.Module):
    def __init__(self, d_embed: int = 64, n_heads: int = 1,
                 n_layers: int = 2, subsample: int = 3, n_test: int = 90, repeat_test: int = 1,
                 no_self: bool = True, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_embed, n_heads, dim_feedforward = 2*d_embed,
                                                        dropout = 0, batch_first = True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        if no_self:
            seq_length = sequence_length(subsample, n_test, repeat_test)
            self.mask = torch.diag(torch.zeros(seq_length) - float('inf')).to(device)
        else:
            self.mask = None

    def forward(self, x):
        return self.encoder(x, mask = self.mask)

class LSTM(nn.Module):
    def __init__(self, d_embed: int = 64, n_test: int = 90, repeat_test: int = 1,
                 n_layers: int = 2, subsample: int = 3):
        super().__init__()
        self.lstm = [UR_LSTM(d_embed, d_embed) for _ in range(n_layers)]

    def forward(self, x):
        for layer in self.lstm:
            x = layer(x)[0]
        return x

class xLSTM(nn.Module):
    def __init__(self, d_embed: int = 64, n_test: int = 90, repeat_test: int = 1,
                 n_layers: int = 2, subsample: int = 3, **kwargs):
        super().__init__()
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    **kwargs
                )
            ),
            context_length=1,
            num_blocks=n_layers,
            embedding_dim=d_embed,
            slstm_at=[],
        )
        self.stack = xLSTMBlockStack(cfg)

    def forward(self, x):
        return self.stack(x)

class Mamba(nn.Module):
    def __init__(self, d_embed: int = 512,
                       n_layers: int = 2,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), **kwargs):
        super().__init__()
        self.mamba = [mamba_ssm.Mamba2(d_model = d_embed,
                                       **kwargs).to(device) for _ in range(n_layers)]

    def forward(self, x):
        for layer in self.mamba:
            x = layer(x)
        return x
