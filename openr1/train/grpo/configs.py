from dataclasses import dataclass, field
from typing import Optional

from openr1.train.grpo.grpo_config import GRPOConfig


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(GRPOConfig):
    """args for callbacks, benchmarks etc."""

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'The benchmarks to run after training.'},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'The callbacks to run during training.'},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The optional system prompt to use for benchmarking.'
        },
    )
    hub_model_revision: Optional[str] = field(
        default='main',
        metadata={'help': 'The Hub model branch to push the model to.'})
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite the Hub revision.'})
    push_to_hub_revision: bool = field(
        default=False,
        metadata={'help': 'Whether to push to a Hub revision/branch.'})
