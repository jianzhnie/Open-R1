from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    r"""
    Configuration class for the [`SFTTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the text field of the dataset. If provided, the trainer will automatically create a
            [`ConstantLengthDataset`] based on `dataset_text_field`.
        packing (`bool`, *optional*, defaults to `False`):
            Controls whether the [`ConstantLengthDataset`] packs the sequences of the dataset.
        max_seq_length (`int` or `None`, *optional*, defaults to `None`):
            Maximum sequence length for the [`ConstantLengthDataset`] and for automatically creating the dataset. If
            `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`.
        dataset_num_proc (`int` or `None`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset. Only used when `packing=False`.
        dataset_batch_size (`Union[int, None]`, *optional*, defaults to `1000`):
            Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is None`,
            tokenizes the full dataset as a single batch.
        dataset_kwargs (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Dictionary of optional keyword arguments to pass when creating packed or non-packed datasets.
        eval_packing (`bool` or `None`, *optional*, defaults to `None`):
            Whether to pack the eval dataset. If `None`, uses the same value as `packing`.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of sequences to use for the [`ConstantLengthDataset`].
        chars_per_token (`float`, *optional*, defaults to `3.6`):
            Number of characters per token to use for the [`ConstantLengthDataset`].
    """
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

    dataset_text_field: str = field(
        default='text',
        metadata={
            'help':
            'Name of the text field of the dataset. If provided, the trainer will automatically create a '
            '`ConstantLengthDataset` based on `dataset_text_field`.'
        },
    )
    packing: bool = field(
        default=False,
        metadata={
            'help':
            'Controls whether the `ConstantLengthDataset` packs the sequences of the dataset.'
        },
    )
    eval_packing: Optional[bool] = field(
        default=None,
        metadata={
            'help':
            'Whether to pack the eval dataset. If `None`, uses the same value as `packing`.'
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'Maximum sequence length for the `ConstantLengthDataset` and for automatically creating the '
            'dataset. If `None`, it uses the smaller value between `tokenizer.model_max_length` and `1024`.'
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=None,
        metadata={
            'help':
            'Number of processes to use for processing the dataset. Only used when `packing=False`.'
        },
    )
    dataset_batch_size: int = field(
        default=1000,
        metadata={
            'help':
            'Number of examples to tokenize per batch. If `dataset_batch_size <= 0` or `dataset_batch_size is '
            'None`, tokenizes the full dataset as a single batch.'
        },
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            'help':
            'Maximum length of the tokenized sequence. Sequences longer than `max_seq_length` are truncated '
            'from the right. If `None`, no truncation is applied. When packing is enabled, this value sets the '
            'sequence length.'
        },
    )
    num_of_sequences: int = field(
        default=1024,
        metadata={
            'help':
            'Number of sequences to use for the `ConstantLengthDataset`.'
        },
    )
    chars_per_token: float = field(
        default=3.6,
        metadata={
            'help':
            'Number of characters per token to use for the `ConstantLengthDataset`.'
        })
    skip_prepare_dataset: bool = field(default=False, )
