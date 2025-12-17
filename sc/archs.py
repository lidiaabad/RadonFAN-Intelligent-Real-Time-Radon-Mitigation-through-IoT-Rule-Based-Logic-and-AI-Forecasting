
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


#Create Serializable module: Extends nn.Module. It adds functionality to save, load, and create modules based on a registry of subclasses

class SerializableModule(nn.Module):

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

# Self-attention computes a weighted sum of all input vectors, where each input 
# has a different weight based on how relevant it is to others. This allows the 
# model to attend to different parts of the sequence

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size) for MultiheadAttention
        attn_output, attn_weights = self.attn(x, x, x)
        return attn_output.transpose(0, 1), attn_weights

# Additive or Bahdanau attention computes a context vector based on the hidden states of 
# the LSTM and attention weights. These weights are learned and reflect how much attention 
# each input should receive.
class BahdanauAttention(nn.Module):
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

    

# Multiplicative or Luong attention calculates the attention scores by using a 
# simple dot product between the query and the key.
class LuongAttention(nn.Module):
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
---------------------------------------------------------------- FINAL ARCHITECTURE PARTS ----------------------------------
'''
    
class AttentionEncoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, forecast_len=6, n_heads=4, step_scale=5.0, short_window=12):
        super().__init__()

        self.hidden_size = hidden_size
        self.forecast_len = forecast_len
        self.n_heads = n_heads
        self.step_scale = step_scale
        self.short_window = short_window

        # --- Step embeddings sinusoidales inicializadas, ligeramente distintas por head ---
        step_emb = torch.zeros(forecast_len, hidden_size)  # [F, H]
        position = torch.arange(0, forecast_len, dtype=torch.float).unsqueeze(1)  # [F, 1]
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0)/hidden_size))  # [H/2]
        step_emb[:, 0::2] = torch.sin(position * div_term)  
        step_emb[:, 1::2] = torch.cos(position * div_term)  
        self.step_emb = nn.Parameter(step_emb)  # [F, H]
        # --- Multihead attention largo y corto ---
        self.attn_long = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
       # self.attn_short = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)

    def forward(self, x_emb, return_attn=False):
        """
        x_emb: [B, T, H]
        returns: context_long, context_short, last_emb
        """
        B, T, H = x_emb.size()

        # --- Positional encoding ---
        pos = torch.arange(T, device=x_emb.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, H, 2, device=x_emb.device) * -(math.log(10000.0)/H))
        pe = torch.zeros(1, T, H, device=x_emb.device)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        x_emb = x_emb + pe

        # --- Queries ---
        last_emb = x_emb[:, -1, :].unsqueeze(1)  # [B,1,H]
        queries = last_emb + self.step_scale * self.step_emb  

        # --- Attention largo ---
        context_long, attn_long = self.attn_long(query=queries, key=x_emb, value=x_emb)

        # --- Attention corto ---
        #short_emb = x_emb[:, -self.short_window:, :]  # últimos window pasos
        #context_short, attn_short = self.attn_short(query=queries, key=short_emb, value=short_emb)

        if return_attn:
            return context_long,last_emb, attn_long
        return context_long, last_emb


class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x: (B, T_in, input_dim)
        out, (h, c) = self.lstm(x)
        return out, (h, c)


class CrossAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_size=1, num_heads=4, dropout=0.3, forecast=6):
        super().__init__()
        self.forecast=forecast
        self.lstm = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4), nn.LayerNorm(hidden_dim//4),
            nn.PReLU(),nn.Dropout(dropout),
            nn.Linear(hidden_dim//4, output_size))

    def forward(self, enc_out, h0, c0, target_seq=None, tf_p=None):

        y_t = enc_out[:, -1:, :]   # keep dim [B,1,D]
        h, c = h0, c0
        outputs = []
        #dec_outs = []
        #prev_y = torch.zeros(enc_out.size(0), 1, device=enc_out.device)
        for t in range(self.forecast):
            query = h[-1].unsqueeze(1)         # [B,1,D]
            attn_out, _ = self.cross_attn(query, enc_out, enc_out)  # [B,1,D]
            dec_in = torch.cat([y_t, attn_out], dim=-1)  # [B,1,2D]
            dec_out, (h, c) = self.lstm(dec_in, (h, c))  # dec_out: [B,1,D]
            #dec_outs.append(dec_out.squeeze(1)) 
            #residual = self.res_proj(dec_in)  # [B,1,D] #LOCALRESIDUAL
            #dec_out = dec_out + residual       # [B,1,D]    
            y_pred = self.reg_head(dec_out.squeeze(1))   # [B,1]
            
            #y_pred = prev_y + y_pred
            #prev_y = y_pred.detach()    
            
            outputs.append(y_pred)
            '''
            if target_seq is not None and np.random.rand() < tf_p:
                y_t = self.proj(target_seq[:, t].unsqueeze(1)).unsqueeze(1)
            else:
                y_t = dec_out
            '''
       # outputs = torch.cat(outputs, dim=1)  # [B, steps]
        #dec_outs = torch.stack(dec_outs, dim=1) # [B, F, H]
        outputs = torch.stack(outputs, dim=1) # [B, steps, n]
        outputs = outputs.squeeze() # por si n=1
        return outputs


class CrossAttentionDecoder1Reg(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.3, forecast=6):
        super().__init__()
        self.forecast=forecast
        self.lstm = nn.LSTM(hidden_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.reg_head = nn.Linear(hidden_dim,1)

    def forward(self, enc_out, h0, c0, target_seq=None, tf_p=None):

        y_t = enc_out[:, -1:, :]   # keep dim [B,1,D]
        h, c = h0, c0
        outputs = []
        dec_outs = []
        for t in range(self.forecast):
            query = h[-1].unsqueeze(1)         # [B,1,D]
            attn_out, _ = self.cross_attn(query, enc_out, enc_out)  # [B,1,D]
            dec_in = torch.cat([y_t, attn_out], dim=-1)  # [B,1,2D]
            
            dec_out, (h, c) = self.lstm(dec_in, (h, c))  # dec_out: [B,1,D]
            dec_outs.append(dec_out.squeeze(1)) 
            
            y_pred = self.reg_head(dec_out.squeeze(1))   # [B,1]
            outputs.append(y_pred)
            '''
            if target_seq is not None and np.random.rand() < tf_p:
                y_t = self.proj(target_seq[:, t].unsqueeze(1)).unsqueeze(1)
            else:
                y_t = dec_out
            '''
        outputs = torch.cat(outputs, dim=1)  # [B, steps]
        return outputs
    

class NoAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads=4, dropout=0.3, forecast=6):
        super().__init__()
        self.forecast=forecast
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2),
            nn.PReLU(),nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1))

    def forward(self, enc_out, h0, c0, target_seq=None, tf_p=None):

        y_t = enc_out[:, -1:, :]   # keep dim [B,1,D]
        h, c = h0, c0
        outputs = []

        #prev_y = torch.zeros(enc_out.size(0), 1, device=enc_out.device)
        for t in range(self.forecast):
            dec_in = y_t
            dec_out, (h, c) = self.lstm(dec_in, (h, c))  # dec_out: [B,1,D   
            y_pred = self.reg_head(dec_out.squeeze(1))   # [B,1]
            outputs.append(y_pred)
            '''
            if target_seq is not None and np.random.rand() < tf_p:
                y_t = self.proj(target_seq[:, t].unsqueeze(1)).unsqueeze(1)
            else:
                y_t = dec_out
            '''
        outputs = torch.cat(outputs, dim=1)  # [B, steps]
        return outputs


class DecoderDirect(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, steps=6, dropout=0.3, forecast_len=6):
        super().__init__()
        self.forecast_len=forecast_len
        self.hidden= hidden_dim
        self.learnable_query = nn.Parameter(torch.randn(1, self.hidden))  # shape (1, D)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2),
            nn.PReLU(),nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1))
    def forward(self, enc_out):
        B = enc_out.size(0)
        # "Decoder" procesa todas las queries en paralelo
        queries = self.learnable_query.unsqueeze(0).expand(B, self.forecast_len, -1)  # [B, steps, D]
        dec_out, _ = self.cross_attn(queries, enc_out, enc_out)  # [B, steps, D]
        y_pred = self.reg_head(dec_out).squeeze(-1)               # [B, steps]d
        return y_pred


class Conv1DClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1D ligera
        self.conv = nn.Sequential(nn.Conv1d(1, 4, kernel_size=3, padding=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(4*12, 8), nn.ReLU(), nn.Linear(8, 1)) 

    def forward(self, x):
        # B, C, L : bacth channels, sequence length: L_out= ((L_in + 2padding - dilation(kernel-1)-1)/stride +1
        # pstride and dilation defect 1: L_out= 12 + 2 - 2- 1 +1 = 12 = L_in
        x = x.unsqueeze(1)  # [batch, 1, 6]
        x = self.conv(x)
        x = self.fc(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # -> [batch, features, seq_len]
        return self.net(x)



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
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()

        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()

        hidden_dim = hidden_size * lookback # tu cálculo original

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) #, 1
        )

        self.cl= nn.Linear(forecast_len, 1)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T

        x_window = x  # ventana inicial
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

            # actualizamos la ventana: movemos 1 y añadimos la predicción
            x_window = torch.cat([x_window[:, :, 1:], out_step.unsqueeze(-1)], dim=2)

        # concatenamos todas las predicciones autoregresivas
        preds = torch.cat(preds, dim=1)  # B x horizon
        cl_out = self.cl(preds)
        
        return preds, cl_out


@SerializableModule.register_model('DC')
class DC(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()

        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()

        hidden_dim = hidden_size * lookback # tu cálculo original

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) #, 1
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T
        
        x = F.pad(x, (self.kernel_size-1, 0)) 
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.prelu1(x)
        #x = self.dropout(x)
        #x = self.pool1(x) 
        
        x = F.pad(x, (2*(self.kernel_size-1), 0)) 
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.prelu2(x)
        #x = self.dropout(x)
        #x = self.pool2(x)
        
        B, _, _ = x.shape
        x = x.view(B, -1)
        #x = self.dropout(x)
        logits = self.fc(x)
        return x, logits


@SerializableModule.register_model('DCRes')
class DCRes(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3,
                 lookback=12, dropout=0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # Bloque 1
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size)
        self.prelu1 = nn.PReLU()

        # Bloque 2
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size)
        self.prelu2 = nn.PReLU()

        # Proyección residual (si input_size != hidden_size)
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

        # ===== Block 1 =====
        x_res1 = self.res_proj(x)
        tmp = F.pad(x, (self.kernel_size-1, 0))
        tmp = self.conv1(tmp)
        tmp = self.prelu1(tmp)
        x = tmp + x_res1                 # Residual connection

        # ===== Block 2 =====
        x_res2 = x                        # same channels, skip ok
        tmp = F.pad(x, (2*(self.kernel_size-1), 0))
        tmp = self.conv2(tmp)
        tmp = self.prelu2(tmp)
        x = tmp + x_res2                 # Segundo residual

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

        # Bloque 1
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.prelu1 = nn.PReLU()

        # Bloque 2
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

        # Block 1
        x = F.pad(x, (self.kernel_size-1, 0))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        # Block 2
        x = F.pad(x, (2*(self.kernel_size-1), 0))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)

        B = x.size(0)
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits
    
    
@SerializableModule.register_model('DC1L')
class DC1L(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()

        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()

        hidden_dim = hidden_size * lookback # tu cálculo original

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1) #, 1
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
    

@SerializableModule.register_model('DC1C')
class DC1C(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()

        hidden_dim = hidden_size * lookback # tu cálculo original
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) #, 1
        )

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(-1)  # B x T x F
        x = x.transpose(1, 2)    # B x F x T
        
        x = F.pad(x, (self.kernel_size-1, 0)) 
        x = self.conv1(x)
        x = self.prelu1(x)

        B, _, _ = x.shape
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits


@SerializableModule.register_model('DC3L')
class DC3L(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()

        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, dilation=2, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()

        hidden_dim = hidden_size * lookback # tu cálculo original

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_size, hidden_size//2),#, 1
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, 1) #, 1
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
    

@SerializableModule.register_model('DC3C')
class DC3C(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size , dilation=2, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv1d(hidden_size, hidden_size , dilation=3, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu3 = nn.PReLU()


        hidden_dim = hidden_size * lookback # tu cálculo original
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) #, 1
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
        
        x = F.pad(x, (3*(self.kernel_size-1), 0)) 
        x = self.conv3(x)
        x = self.prelu3(x)

        B, _, _ = x.shape
        x = x.view(B, -1)
        logits = self.fc(x)
        return x, logits
    

@SerializableModule.register_model('DCMP')
class DCMP(SerializableModule):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, forecast_len=3, lookback=12, dropout= 0.3, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # First convolutional block
        self.conv1 = nn.Conv1d(input_size, hidden_size , kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size ) #, padding=kernel_size-1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        hidden_dim = hidden_size * lookback//3 # tu cálculo original

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1) #, 1
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
    