import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout):
        super(Transformer, self).__init__()
        # n_embed = 15
        # dim_embed = round(n_embed**0.25)

        # Layers
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=200)

        # self.embed = nn.Embedding(num_embeddings=n_embed, embedding_dim=dim_embed)
        self.linear_src = nn.Linear(src_dim, d_model)
        # self.linear_src = nn.Linear(src_dim + dim_embed - 1, d_model)
        self.linear_tgt = nn.Linear(tgt_dim + 3, d_model)

        ''' SDDP-Transformer'''
        # self.transformer = nn.Transformer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        ''' SDDP-Transformer (Decoder)'''
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.linear_out = nn.Linear(d_model, tgt_dim + 3)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # src, tgt size = (batch_size, sequence length)

        # fc + positional encoding => output size = (batch_size, sequence length, d_model)
        # stage_embed = self.embed(torch.mul(src[:, :, -1], self.n_stage-1).long())
        token_encoding = F.one_hot(tgt[:, :, -1].to(torch.long), num_classes=4)
        tgt = torch.concat((tgt[:, :, :-1], token_encoding), dim=2)
        # stage_embed = self.embed(src[:, :, -1].long())
        # src = torch.concat((src[:, :, :-1], stage_embed), dim=2)
        # src = torch.flatten(src, start_dim=1)
        src = self.linear_src(src) # self.linear_src(src[:, :, :-1])

        # src = self.positional_encoding(src)
        tgt = self.linear_tgt(tgt)
        tgt = self.positional_encoding(tgt)

        ''' SDDP-Transformer'''
        # transformer_out, encoder_weights, decoder_weights_sa, decoder_weights_mha = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # out = self.linear_out(transformer_out)

        ''' SDDP-Transformer (Decoder)'''
        decoder_out, decoder_weights_sa, decoder_weights_mha = self.decoder(tgt, src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.linear_out(decoder_out)

        return out, [], decoder_weights_sa, decoder_weights_mha # out, encoder_weights, decoder_weights_sa, decoder_weights_mha

    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1).float() # Lower triangular matrix
        mask = mask.masked_fill(mask == 0, float('-inf')) # convert zeros(false) to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # convert ones to 0

        return mask

    def get_pad_mask(self, max_seq_len, data) -> torch.tensor:
        pad_mask = torch.ones((len(data), max_seq_len))
        for idx in range(len(data)):
            sample_seq_len = data[idx].shape[0]-1
            pad_mask[idx, :sample_seq_len] = torch.zeros(sample_seq_len)
        pad_mask = pad_mask == 1
        return pad_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding.requires_grad = False
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # saving buffer (same as parameter without gradients needed)
        # optimizer가 업데이트하지 않고 하나의 layer로써 작동. GPU 연산 가능
        self.pos_encoding = pos_encoding.unsqueeze(0).to("cuda")

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        _, seq_len, _ = token_embedding.size()
        pos_embed = self.pos_encoding[:, :seq_len, :]
        out = token_embedding + pos_embed
        # Residual connection + pos encoding
        return out