from src.library import *


class FixedPositionalEncoding(nn.Module):
    """Positional encoding of input.
    Args:
        d_model: the embed dim.
        dropout: the dropout value.
        max_len: the max. length of the incoming sequence.
        scale_factor: the scale factor for the positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        scale_factor: float = 1.0,
    ):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x: Tensor) -> Tensor:
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Positional encoding of input. This is learnable.

    Args:
        d_model: the embed dim.
        dropout: the dropout value.
        max_len: the max. length of the incoming sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model.
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding: str) -> Type[nn.Module]:
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )
