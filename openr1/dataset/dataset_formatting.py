import logging
from typing import Callable, Literal, Optional, Union

from datasets import Dataset, Value
from transformers import PreTrainedTokenizer
"""
Brow code from huggingface trl
"""

FORMAT_MAPPING = {
    'chatml': [{
        'content': Value(dtype='string', id=None),
        'role': Value(dtype='string', id=None)
    }],
    'instruction': {
        'completion': Value(dtype='string', id=None),
        'prompt': Value(dtype='string', id=None)
    },
}


def conversations_formatting_function(
    tokenizer: PreTrainedTokenizer,
    messages_field: Literal['messages', 'conversations'],
    tools: Optional[list] = None,
):
    r"""
    return a callable function that takes in a "messages" dataset and returns a formatted dataset,
    based on the tokenizer apply chat template to the dataset along with the schema of the list of
    functions in the tools list.
    """

    def format_dataset(examples):  # -> list | Any:
        if isinstance(examples[messages_field][0], list):
            output_texts = []
            for i in range(len(examples[messages_field])):
                output_texts.append(
                    tokenizer.apply_chat_template(examples[messages_field][i],
                                                  tokenize=False,
                                                  tools=tools))
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples[messages_field],
                                                 tokenize=False,
                                                 tools=tools)

    return format_dataset


def instructions_formatting_function(tokenizer: PreTrainedTokenizer):
    r"""
    return a callable function that takes in an "instructions" dataset and returns a formatted dataset,
    based on the tokenizer apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples['prompt'], list):
            output_texts = []
            for i in range(len(examples['prompt'])):
                converted_sample = [
                    {
                        'role': 'user',
                        'content': examples['prompt'][i]
                    },
                    {
                        'role': 'assistant',
                        'content': examples['completion'][i]
                    },
                ]
                output_texts.append(
                    tokenizer.apply_chat_template(converted_sample,
                                                  tokenize=False))
            return output_texts
        else:
            converted_sample = [
                {
                    'role': 'user',
                    'content': examples['prompt']
                },
                {
                    'role': 'assistant',
                    'content': examples['completion']
                },
            ]
            return tokenizer.apply_chat_template(converted_sample,
                                                 tokenize=False)

    return format_dataset


def get_formatting_func_from_dataset(
        dataset: Union[Dataset],
        tokenizer: PreTrainedTokenizer,
        tools: Optional[list] = None) -> Optional[Callable]:
    r"""
    Finds the correct formatting function based on the dataset structure. Currently supported datasets are:
    - `ChatML` with [{"role": str, "content": str}]
    - `instruction` with [{"prompt": str, "completion": str}]

    Args:
        dataset (Dataset): User dataset
        tokenizer (PreTrainedTokenizer): Tokenizer used for formatting

    Returns:
        Callable: Formatting function if the dataset format is supported else None
    """
    if isinstance(dataset, Dataset):
        if 'messages' in dataset.features:
            if dataset.features['messages'] == FORMAT_MAPPING['chatml']:
                logging.info('Formatting dataset with chatml format')
                return conversations_formatting_function(
                    tokenizer, 'messages', tools)
        if 'conversations' in dataset.features:
            if dataset.features['conversations'] == FORMAT_MAPPING['chatml']:
                logging.info('Formatting dataset with chatml format')
                return conversations_formatting_function(
                    tokenizer, 'conversations', tools)
        elif dataset.features == FORMAT_MAPPING['instruction']:
            logging.info('Formatting dataset with instruction format')
            return instructions_formatting_function(tokenizer)

    return None


def pack_examples(examples: dict[str, list[list]],
                  seq_length: int) -> dict[str, list[list]]:
    """Pack examples into chunks of size `seq_length`.

    Args:
        examples (`dict[str, list[list]]`):
            Dictionary of examples with keys as strings and values as lists of lists.
        seq_length (`int`):
            Maximum sequence length.

    Returns:
        `dict[str, list[list]]`: Dictionary of examples with keys as strings and values as lists of lists.

    Example:

    ```python
    >>> from trl import pack_examples
    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> pack_examples(examples, seq_length=5)
    {'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8]], 'attention_mask': [[0, 1, 1, 0, 0], [1, 1, 1]]}
    >>> pack_examples(examples, seq_length=2)
    {'input_ids': [[1, 2], [3, 4], [5, 6], [7, 8]], 'attention_mask': [[0, 1], [1, 0], [0, 1], [1, 1]]}
    ```
    """
    # Join  all the values into a single list
    examples = {k: sum(v, []) for k, v in examples.items()}
    # Split the values into chunks of size seq_length
    examples = {
        k: [v[i:i + seq_length] for i in range(0, len(v), seq_length)]
        for k, v in examples.items()
    }
    return examples
