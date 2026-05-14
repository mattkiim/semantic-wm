import torch
from ..base_autoencoder import BaseAutoencoder


class PrecomputedEncoder(BaseAutoencoder):
    """Pass-through encoder for pre-computed backbone features.

    Used when the dataset already contains patch embeddings (e.g. DINO
    features from combined_v3 HDF5 files).  ``encode()`` returns the input
    unchanged; there is no pixel decoder.

    Set ``has_decoder = False`` so adapter validation skips pixel-space metrics.
    """

    has_decoder = False

    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__()
        self._latent_dim = embedding_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PrecomputedEncoder has no pixel decoder")
