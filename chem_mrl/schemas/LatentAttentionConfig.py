from dataclasses import asdict, dataclass

from chem_mrl.constants import BASE_MODEL_HIDDEN_DIM


@dataclass
class LatentAttentionConfig:
    hidden_dim: int = BASE_MODEL_HIDDEN_DIM
    num_latents: int = 512
    num_cross_heads: int = 8
    cross_head_dim: int = 64
    output_normalize: bool = True
    enable: bool = True
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.hidden_dim, int):
            raise TypeError("hidden_dim must be an int")
        if not isinstance(self.num_latents, int):
            raise TypeError("num_latents must be an int")
        if not isinstance(self.num_cross_heads, int):
            raise TypeError("num_cross_heads must be an int")
        if not isinstance(self.cross_head_dim, int):
            raise TypeError("cross_head_dim must be an int")
        if not isinstance(self.output_normalize, bool):
            raise TypeError("output_normalize must be a bool")
        if not isinstance(self.enable, bool):
            raise TypeError("enable must be a bool")
        # check values
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if self.num_cross_heads <= 0:
            raise ValueError("num_cross_heads must be positive")
        if self.cross_head_dim <= 0:
            raise ValueError("cross_head_dim must be positive")
