import torch
from torch import nn

from model.encoder_decoder.mlp_module import Encoder, Decoder
from model.encoder_decoder.utils import correlation_score


class EncoderDecoder(nn.Module):
    def __init__(self, opt):

        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(
                input_dim=opt.enc_h_dim,
                output_dim=opt.enc_h_dim,
                n_blk=opt.enc_n_blk,
                h_dim=opt.enc_h_dim,
                skip=opt.skip,
                dropout_p=opt.enc_dropout_p,
                activation=opt.activation,
                norm=opt.norm,
        )

        self.decoder = Decoder(
                input_dim=opt.enc_h_dim,
                output_dim=opt.output_dim,
                n_blk=opt.dec_n_blk,
                h_dim=opt.dec_h_dim,
                skip=opt.skip,
                dropout_p=opt.dec_dropout_p,
                activation=opt.activation,
                norm=opt.norm,
        )

        self.encoder_in_fc = nn.Linear(opt.input_dim, opt.encoder_h_dim)

        decoder_out_fc = []
        decoder_out_res_fc = []
        for _ in range(n_dec_blk + 1):
            decoder_out_fc.append(nn.Linear(opt.dec_h_dim, opt.output_dim))
            decoder_out_res_fc.append(nn.Linear(opt.dec_h_dim, opt.output_dim))
        self.decoder_out_fc = nn.ModuleList(decoder_out_fc)
        self.decoder_out_res_fc = nn.ModuleList(decoder_out_res_fc)

    def _encode(self, x):
        # x: (B,445/419)
        h = self.encoder_in_fc(x)
        z = self.encoder(h)
        return z # (B, enc_h_dim)

    def _decode(self, z):
        h = z
        output, _ = self.decoder(h)
        return output

    def forward(self, x):
        # x: (B,445/419)
        z = self._encode(x)
        output = self._decode(z)
        return output # (B, 5)
