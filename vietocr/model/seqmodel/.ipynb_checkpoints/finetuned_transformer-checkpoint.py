from einops import rearrange
from torchvision import models
import math
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from typing import Optional, Any, Union, Callable

from vietocr.model.backbone.cnn import CNN

class FinetunedTransformer(nn.Module):
    def __init__(self, vocab_size, backbone, cnn_args, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, 
                 max_seq_length: int=1024, pos_dropout: float = 0.1,
                 trans_dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 finetuned_encoder: Optional[Any] = None, finetuned_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FinetunedTransformer, self).__init__()

        if finetuned_encoder is not None:
            self.encoder = finetuned_encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, trans_dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = FinetunedEncoder(backbone, cnn_args, d_model, pos_dropout, max_seq_length, encoder_layer, num_encoder_layers, encoder_norm)

        if finetuned_decoder is not None:
            self.decoder = finetuned_decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, trans_dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = FinetunedDecoder(d_model, pos_dropout, max_seq_length, vocab_size, decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first
        
        
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, img: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        memory = self.encoder(img, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output



class FinetunedEncoder(nn.TransformerEncoder):
    def __init__(self, backbone, cnn_args, d_model, pos_dropout, max_seq_length, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.cnn = CNN(backbone, **cnn_args)

        self.d_model = d_model


    def forward(self, img):
        src = self.cnn(img)
        src = self.pos_enc(src*math.sqrt(self.d_model))
        memory = super().forward(src)
        
        return memory.transpose(0,1)

class FinetunedDecoder(nn.TransformerDecoder):
    def __init__(self, d_model, pos_dropout, max_seq_length, vocab_size, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.fc = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

        

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
       
        return mask


    def forward(self, tgt, memory, tgt_key_padding_mask=None):
        memory = memory.transpose(0,1)
        tgt = tgt.transpose(0, 1)
        tgt_mask = self.gen_nopeek_mask(tgt.size(0)).to(tgt.device)
        tgt = self.embed_tgt(tgt)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model))
        output = super().forward(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = output.transpose(0, 1)

        return self.fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)