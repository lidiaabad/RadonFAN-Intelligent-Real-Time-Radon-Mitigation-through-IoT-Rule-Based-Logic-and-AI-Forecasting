
# IMPORTS 
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
import copy
import math
import random
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm


class SerializableModule(nn.Module):
    #Create Serializable module: Extends nn.Module. It adds functionality to save, load, and create modules based on a registry of subclasses
    subclasses = {}
    def __init__(self):
        super().__init__()

    # Registers a subclass with a given name in the  subclasses dictionary
    @classmethod
    def register_model(cls, model_name):
        def decorator(subclass):
            cls.subclasses[model_name] = subclass
            return subclass

        return decorator
    
    # Instantiate a registered subclass
    @classmethod
    def create(cls, arc, **kwargs):
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc](**kwargs)

    #Saves the module's state dictionary (parameters) to a file.
    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    #save the architecture and parameters
    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    #load the state dictionary from a file into an instance
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

'''
------------------------------------------------------- HELPERS ---------------------------------------
'''

### Attention

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size) for MultiheadAttention
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_output.transpose(0, 1), attn_weights

class BahdanauAttention(nn.Module):
    # Additive or Bahdanau attention 
    def __init__(self, hidden_size):
        super().__init__()
        self.attn_enc = nn.Linear(hidden_size, hidden_size)
        self.attn_dec = nn.Linear(hidden_size, hidden_size)
        self.attn_score = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [B, L_enc, H], decoder_hidden: [B, H]
        dec_exp = self.attn_dec(decoder_hidden).unsqueeze(1)  # [B, 1, H]
        enc_proj = self.attn_enc(encoder_outputs)             # [B, L_enc, H]
        scores = self.attn_score(torch.tanh(enc_proj + dec_exp)) * 5# [B, L_enc, 1]
        attn_weights = F.softmax(scores, dim=1)              # [B, L_enc, 1]
        context = (attn_weights * encoder_outputs).sum(dim=1)     # [B, H]
        return context, attn_weights

    
class LuongAttention(nn.Module):
    # Multiplicative or Luong attention
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, lstm_output, encoder_output):
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(1)  # [B, 1, H]
        if lstm_output.dim() == 2:
            lstm_output = lstm_output.unsqueeze(1)       # [B, 1, H]
        # [B, T_enc, H] -> [B, H, T_enc]
        encoder_proj = self.attn(encoder_output).transpose(1, 2)
        # bmm: [B, T_dec, H] @ [B, H, T_enc] -> [B, T_dec, T_enc]
        attn_weights = torch.bmm(lstm_output, encoder_proj)
        attn_weights = torch.softmax(attn_weights, dim=2)
        # context: [B, T_dec, T_enc] @ [B, T_enc, H] -> [B, T_dec, H]
        context = torch.bmm(attn_weights, encoder_output)
        return context, attn_weights



'''
----------------------------------------------------------- DL  BASELINE -----------------------------------
'''

    
@SerializableModule.register_model('R2C')
class R2C(SerializableModule):
    def __init__(self, lookback=36, forecast_len=6, in_channels=1, n_filters=32, kernel_size=3, 
                 n_blocks=2, input_size=1, hidden_size=32, num_layers=2, dropout=0.3):
        super().__init__()

        self.kernel_size = kernel_size
        self.forecast_len = forecast_len
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) 
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size )
        self.prelu2 = nn.PReLU()
        hidden_dim = hidden_size * lookback
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) 
        )
        self.cl= nn.Linear(forecast_len, 1)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T

        x_window = x 
        preds = []

        for _ in range(self.forecast_len):# self.forecast_len
            tmp = F.pad(x_window, (self.kernel_size - 1, 0))  # pad left
            tmp = self.conv1(tmp)
            tmp = self.prelu1(tmp)
            tmp = F.pad(tmp, (2*(self.kernel_size - 1), 0))  # pad left
            tmp = self.conv2(tmp)
            tmp = self.prelu2(tmp)
            B, _, _ = tmp.shape
            tmp= tmp.view(B, -1)
            out_step = self.fc(tmp)
            preds.append(out_step)
            x_window = torch.cat([x_window[:, :, 1:], out_step.unsqueeze(-1)], dim=2)

        preds = torch.cat(preds, dim=1)  # B x fc
        cl_out = self.cl(preds)
        
        return preds, cl_out


@SerializableModule.register_model('DC')
class DC(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size )
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size )
        self.prelu2 = nn.PReLU()
        hidden_dim = hidden_size * lookback 
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) 
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T
        
        x = F.pad(x, (self.kernel_size-1, 0)) 
        x = self.conv1(x)
        x = self.prelu1(x)
        x = F.pad(x, (2*(self.kernel_size-1), 0)) 
        x = self.conv2(x)
        x = self.prelu2(x)
        B, _, _ = x.shape
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits


@SerializableModule.register_model('DCRes')
class DCRes(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3,
                 lookback=12, dropout=0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size)
        self.prelu2 = nn.PReLU()
        self.res_proj = (
            nn.Conv1d(input_size, hidden_size, kernel_size=1)
            if input_size != hidden_size else nn.Identity()
        )
        hidden_dim = hidden_size * lookback
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)
        x_res1 = self.res_proj(x)
        tmp = F.pad(x, (self.kernel_size-1, 0))
        tmp = self.conv1(tmp)
        tmp = self.prelu1(tmp)
        x = tmp + x_res1                 # Residual connection
        x_res2 = x                       
        tmp = F.pad(x, (2*(self.kernel_size-1), 0))
        tmp = self.conv2(tmp)
        tmp = self.prelu2(tmp)
        x = tmp + x_res2                # Residual connection
        B = x.size(0)
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits

@SerializableModule.register_model('DCBN')
class DCBN(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3,
                 lookback=12, dropout=0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size) # BN
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.prelu2 = nn.PReLU()
        hidden_dim = hidden_size * lookback
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size-1, 0))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = F.pad(x, (2*(self.kernel_size-1), 0))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        B = x.size(0)
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits


@SerializableModule.register_model('DCMP')
class DCMP(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) 
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)                                     # MP
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size ) 
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)                                     # MP
        hidden_dim = hidden_size * lookback//3 
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) 
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T
        x = F.pad(x, (self.kernel_size-1, 0)) 
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x) 
        x = F.pad(x, (2*(self.kernel_size-1), 0)) 
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        B, _, _ = x.shape
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits

    
