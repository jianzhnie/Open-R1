import math
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import transformers
from transformers import (DataCollatorForLanguageModeling, PreTrainedModel,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

sys.path.append(os.getcwd())
from openr1.dataset.pretrain_dataset import PretrainDataset
from openr1.utils.logger_utils import get_logger

logger = get_logger('openr1')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn.'
        },
    )
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
    train_data_split: str = field(default=None,
                                  metadata={'help': 'Dataset split.'})
    eval_data_path: str = field(
        default=None, metadata={'help': 'Path to the evaluation data.'})
    eval_data_split: str = field(default=None,
                                 metadata={'help': 'Dataset split.'})
    data_cache_dir: str = field(
        default=None, metadata={'help': 'Path to the cache the data.'})


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=512,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )


def trainer_save_model_safe(trainer: Trainer) -> None:
    """Safely save model when using FSDP (Fully Sharded Data Parallel).

    Args:
        trainer: The Huggingface trainer instance
    """
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


def make_pretrain_data_module(data_args: DataArguments,
                              training_args: TrainingArguments,
                              tokenizer: PreTrainedTokenizer):
    """Create training and evaluation datasets.

    Args:
        data_args: Dataset configuration arguments
        training_args: Training configuration arguments
        tokenizer: The tokenizer to use for processing text

    Returns:
        Dictionary containing train and eval datasets
    """
    train_dataset = PretrainDataset(data_path=data_args.data_path,
                                    split=data_args.train_data_split,
                                    cache_dir=data_args.data_cache_dir,
                                    tokenizer=tokenizer,
                                    max_length=training_args.model_max_length)

    if data_args.eval_data_path:
        eval_dataset = PretrainDataset(
            data_path=data_args.eval_data_path,
            split=data_args.eval_data_split,
            cache_dir=data_args.data_cache_dir,
            tokenizer=tokenizer,
            max_length=training_args.model_max_length)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train() -> None:
    """Main training function that orchestrates the entire training process."""

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model and tokenizer
    logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_tokenizer(model_args, training_args)
    logger.info('Successfully loaded model and tokenizer.')

    # Create a dataset and Trainer, then train the model
    logger.info('Creating a dataset and DataCollator...')
    dataset_module = make_pretrain_data_module(data_args,
                                               training_args,
                                               tokenizer=tokenizer)
    logger.info('Successfully created the dataset.')
    logger.info('Creating DataCollator...')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        **dataset_module,
    )

    # Training
    if training_args.do_train:
        checkpoint_path = pathlib.Path(training_args.output_dir)
        has_checkpoints = list(checkpoint_path.glob('checkpoint-*'))

        if has_checkpoints and training_args.resume_from_checkpoint:
            logger.info('Resuming training from checkpoint %s',
                        training_args.resume_from_checkpoint)
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            logger.info('Starting training from scratch...')
            train_result = trainer.train()

        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)

        # Save model
        trainer.model.config.use_cache = True
        trainer.save_state()
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix='eval')
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')

        metrics['perplexity'] = perplexity
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    logger.info('Done.')


if __name__ == '__main__':
    train()
