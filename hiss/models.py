import math
import hydra
import torch
import torch.nn as nn

try:
    from hiss.s4 import S4Block as S4
except ImportError:
    pass

try:
    from mamba_ssm.models.mixer_seq_simple import create_block
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("Failed to import Triton LayerNorm / RMSNorm kernels")


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lstm_hidden_dim,
        nlayers,
        dropout,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_type = "lstm"
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            lstm_hidden_dim,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        embed = self.embed(x)
        out, _ = self.lstm(embed.reshape((-1, *embed.shape[-2:])))
        out = self.decoder(out.reshape((*x.shape[:-2], *out.shape[-2:])))
        return out


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim,
        nheads,
        nlayers,
        dropout=0.5,
        prenorm=False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(model_dim, max_len=50000)  # , dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=nheads,
                dim_feedforward=model_dim,
                dropout=dropout,
                norm_first=prenorm,
                batch_first=True,
            ),
            num_layers=nlayers,
        )

        self.embed = nn.Linear(input_dim, model_dim)
        self.decoder = nn.Linear(model_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            # Add causal masking
            device = src.device
            if self.src_mask is None or self.src_mask.size(1) != src.size(1):
                mask = nn.Transformer.generate_square_subsequent_mask(
                    src.size(1), device
                )
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.embed(src)
        src = self.pos_encoder(src)

        output = self.transformer(src, mask=self.src_mask, is_causal=has_mask)

        output = self.decoder(output)
        return output


class S4Net(nn.Module):
    def __init__(
        self,
        model_dim,
        dropout,
        nlayers,
        prenorm,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.s4_layers.append(S4(model_dim, dropout=dropout, transposed=False))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, lengths=None):
        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        return x


class S4Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim,
        nlayers=4,
        dropout=0.2,
        prenorm=False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.model_type = "s4_model"
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.embed = nn.Linear(input_dim, model_dim)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(nlayers):
            self.s4_layers.append(S4(model_dim, dropout=dropout, transposed=False))
            self.norms.append(nn.LayerNorm(model_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, x, lengths=None):
        """
        Input x is shape (B, L, input_dim)
        """
        x = self.embed(x)  # (B, L, input_dim) -> (B, L, hidden_dim)

        x = x.transpose(-1, -2)  # (B, L, hidden_dim) -> (B, hidden_dim, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, hidden_dim, L) -> (B, hidden_dim, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, lengths=lengths)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, hidden_dim) -> (B, output_dim)

        return x


class MambaNet(nn.Module):
    def __init__(
        self,
        model_dim,
        nlayers,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    model_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(nlayers)
            ]
        )
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            model_dim, eps=norm_epsilon
        )

    def forward(self, hidden_states, lengths=None, inference_params=None):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim=128,
        nlayers=1,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        fused_add_norm=False,
        residual_in_fp32=False,
    ) -> None:
        super().__init__()
        self.model_type = "mamba"
        self.embed = nn.Linear(input_dim, model_dim)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.mamba = MambaNet(
            model_dim=model_dim,
            nlayers=nlayers,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )

        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, x, inference_params=None):
        hidden_states = self.embed(x)
        hidden_states = self.mamba(hidden_states)
        output = self.decoder(hidden_states)
        return output


class LowFreqPredictor(nn.Module):
    def __init__(self, model, pred_freq_ratio):
        super().__init__()
        self.model = model
        self.model_type = model.model_type
        self.pred_freq_ratio = pred_freq_ratio

    def forward(self, x):
        pred = self.model(x)
        return pred[:, :: self.pred_freq_ratio]


class HierarchicalModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim,
        ll_model,
        hl_model,
        freq_ratio=10.0,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.model_type = "hierarchical"
        self.low_level = hydra.utils.instantiate(ll_model)(input_dim=input_dim)
        self.high_level = hydra.utils.instantiate(hl_model)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(model_dim)  # , dropout)
        self.output = nn.Linear(model_dim, output_dim)
        self.freq_ratio = freq_ratio

    def forward(self, x, has_mask=True):
        # Reshape x into B, Lc, C, D
        x = x.reshape(x.shape[0], -1, int(self.freq_ratio), x.shape[-1])

        proc_x = self.low_level(x.reshape(-1, *x.shape[2:]))[:, -1]

        proc_x = proc_x.reshape(*x.shape[:2], -1)

        if isinstance(self.high_level, nn.TransformerEncoder):
            # Pass through transformer to output B, Lc, D
            if has_mask:
                device = x.device
                if self.src_mask is None or self.src_mask.size(1) != proc_x.size(1):
                    mask = nn.Transformer.generate_square_subsequent_mask(
                        proc_x.size(1), device
                    )
                    self.src_mask = mask
            else:
                self.src_mask = None

            proc_x = self.pos_encoder(proc_x)
            output = self.high_level(proc_x, mask=self.src_mask, is_causal=has_mask)
        else:
            output = self.high_level(proc_x)
        output = self.output(output)
        return output
