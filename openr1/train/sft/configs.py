from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class SFTConfig(trl.SFTConfig):
    """args for callbacks, benchmarks etc."""

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'The benchmarks to run after training.'})
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'The callbacks to run during training.'})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The optional system prompt to use for benchmarking.'
        },
    )
    hub_model_revision: Optional[str] = field(
        default='main',
        metadata={'help': 'The Hub model branch to push the model to.'},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={'help': 'Whether to overwrite the Hub revision.'})
    push_to_hub_revision: bool = field(
        default=False,
        metadata={'help': 'Whether to push to a Hub revision/branch.'})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={'help': ('The entity to store runs under.')},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={'help': ('The project to store runs under.')},
    )
