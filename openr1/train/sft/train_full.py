import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import transformers
from datasets import load_dataset
from transformers import (AutoTokenizer, HfArgumentParser, PreTrainedModel,
                          PreTrainedTokenizer, TrainingArguments)
from transformers.trainer_pt_utils import LabelSmoother

sys.path.append(os.getcwd())
from openr1.dataset.supervised_dataset import sft_prepare_dataset
from openr1.utils.logger_utils import get_logger

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

logger = get_logger('openr1')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
    trust_remote_code: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to allow for custom models defined on the Hub in their own modeling files'
        },
    )
    padding_side: str = field(
        default='right', metadata={'help': 'The padding side in tokenizer'})


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})
    eval_data_path: str = field(
        default=None, metadata={'help': 'Path to the evaluation data.'})


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=2048,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT,
                              save_policy):
        trainer.save_model()


def load_model_tokenizer(
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load and configure the model and tokenizer.

    Args:
        model_args: Model configuration arguments
        training_args: Training configuration arguments

    Returns:
        Tuple containing the loaded model and tokenizer
    """

    config_kwargs = {
        'cache_dir': model_args.cache_dir,
        'trust_remote_code': model_args.trust_remote_code,
    }
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, **config_kwargs)
    orig_ctx_len = getattr(config, 'max_position_embeddings', None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {'type': 'linear', 'factor': scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config, **config_kwargs)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        **config_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Enable model parallelism
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.config.use_cache = (
            False  # Turn off when gradient checkpointing is enabled
        )
    return model, tokenizer


def train() -> None:
    """Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.

    Returns:
        None
    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if processing_class is None:
        processing_class = AutoTokenizer.from_pretrained(
            model.config._name_or_path)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token  # required for padding when collating data

    formatting_func = None

    train_dataset = load_dataset('simplescaling/s1K')
    train_dataset = sft_prepare_dataset(train_dataset, processing_class,
                                        formatting_func, data_args)
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            eval_dataset = {
                key: sft_prepare_dataset(dataset, processing_class,
                                         formatting_func, data_args)
                for key, dataset in eval_dataset.items()
            }
        else:
            eval_dataset = sft_prepare_dataset(eval_dataset, processing_class,
                                               formatting_func, data_args)

    # Load model and tokenizer
    logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(model_args, training_args)
    logger.info('Successfully loaded model and tokenizer.')

    # Create the training dataset and data collator

    # Initialize the Trainer object and start training
    logging.warning('Instantiating Trainer')
    logging.warning('Done.')


if __name__ == '__main__':
    train()
