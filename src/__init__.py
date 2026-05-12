"""src — top-level package for the world-model-rae codebase.

Subpackages:
  models/   – autoencoders, adapters, DiT, world model
  training/ – training loops, diffusion, validation
  data/     – dataset loaders
  utils/    – evaluation helpers
"""

from .models.world_model import WorldModel
from .models.base_autoencoder import (
    BaseAutoencoder,
    create_autoencoder,
    encoder_config_from_args,
)
from .models.encoders.vae import VAE
from .models.encoders.rae import RAE
from .models.encoders.scale_rae import ScaleRAE
from .models.adapters import (
    BaseAdapter,
    IdentityAdapter,
    MLPAdapter,
    SVAEAdapter,
    create_adapter,
    adapter_config_from_args,
)
from .training.train import train_wm
from .training.train_adapter import train_adapter
from .utils.utils import (
    aggregate_model_results,
    discover_trials,
    predict,
    rescale_bridge_action,
)

__all__ = [
    "WorldModel",
    "BaseAutoencoder",
    "create_autoencoder",
    "encoder_config_from_args",
    "VAE",
    "RAE",
    "ScaleRAE",
    "BaseAdapter",
    "IdentityAdapter",
    "MLPAdapter",
    "SVAEAdapter",
    "create_adapter",
    "adapter_config_from_args",
    "train_wm",
    "train_adapter",
    "aggregate_model_results",
    "discover_trials",
    "predict",
    "rescale_bridge_action",
]
