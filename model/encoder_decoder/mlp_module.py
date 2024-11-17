from torch import nn


class LinearBlock(nn.Module):
    def __init__(self, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="none"):
        super(LinearBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(h_dim, h_dim, bias=False)
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        elif norm == "layer_norm":
            self.norm = nn.LayerNorm(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        else:
            self.norm = None
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        if activation == "relu":
            self.activ = nn.ReLU()
        elif activation == "gelu":
            # print("activation", activation)
            self.activ = nn.GELU()
        else:
            raise RuntimeError()

    def forward(self, x):
        h = x
        h_prev = x
        h = self.activ(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc(h)
        if self.skip:
            h = h + h_prev
        return h


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_blk, h_dim=128, skip=False, 
                    dropout_p=0.1, activation="relu", norm="bn",):

        super(Encoder, self).__init__()
        layers = []
        for _ in range(n_blk):
            layers.append(LinearBlock(input_dim=input_dim, h_dim=h_dim, 
                                      skip=skip, dropout_p=dropout_p, 
                                      activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        # x: (B,enc_h_dim)
        h = x
        for layer in self.layers:
            h = layer(h)
        z = self.out_fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_blk, h_dim=128, skip=False, 
                    dropout_p=0.1, activation="relu", norm="bn",):

        super(Decoder, self).__init__()

        self.in_fc = nn.Linear(input_dim, h_dim)
        layers = []
        for _ in range(n_blk):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h = x
        h = self.in_fc(h)
        hs = [h] # len: 1 + n_blk, 每一层layer的output
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        output = self.out_fc(h)
        return output, hs
