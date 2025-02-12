import warnings
from typing import Callable, Dict, Optional, Union

import datasets
import torch
from accelerate import PartialState
from datasets import Dataset, IterableDataset, load_dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import (BaseImageProcessor, FeatureExtractionMixin,
                          PreTrainedTokenizer, PreTrainedTokenizerBase,
                          ProcessorMixin)
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.extras.dataset_formatting import ConstantLengthDataset

from openr1.configs.data_args import DataArguments
from openr1.dataset.dataset_formatting import pack_examples
from openr1.utils.logger_utils import get_logger

logger = get_logger('openr1')


def sft_prepare_dataset(
    dataset: Union[Dataset, IterableDataset],
    processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor,
                            FeatureExtractionMixin, ProcessorMixin],
    formatting_func: Optional[Callable[[dict], str]],
    data_args: DataArguments = None,
) -> Union[Dataset, IterableDataset]:
    # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
    if isinstance(dataset, ConstantLengthDataset):
        return dataset

    # Build the kwargs for the `map` function
    map_kwargs = {}
    if isinstance(dataset, Dataset):
        # IterableDataset does not support num_proc
        map_kwargs['num_proc'] = data_args.dataset_num_proc

    with PartialState().local_main_process_first():
        # Apply the formatting function if any
        if formatting_func is not None:
            if isinstance(dataset, Dataset):
                # `IterableDataset.map` does not support `desc`
                map_kwargs[
                    'desc'] = f'Applying formatting function to {data_args.data_path} dataset'

            batched = isinstance(formatting_func(next(iter(dataset))), list)

            def _func(example):
                return {'text': formatting_func(example)}

            dataset = dataset.map(_func, batched=batched, **map_kwargs)

        # If the dataset is prompt-completion, convert it to language modeling type
        if 'prompt' in dataset.column_names and 'completion' in dataset.column_names:
            key = 'messages' if is_conversational(dataset[0]) else 'text'

            def concat_prompt_completion(example):
                return {key: example['prompt'] + example['completion']}

            dataset = dataset.map(concat_prompt_completion,
                                  remove_columns=['prompt', 'completion'])

        # Apply the chat template if needed
        if isinstance(dataset, Dataset):
            # `IterableDataset.map` does not support `desc`
            map_kwargs[
                'desc'] = f'Applying chat template to {data_args.data_path} dataset'
        dataset = dataset.map(
            maybe_apply_chat_template,
            fn_kwargs={'tokenizer': processing_class},
            remove_columns='messages'
            if 'messages' in dataset.column_names else None,
            **map_kwargs,
        )

        # Tokenize the dataset
        if isinstance(dataset, Dataset):
            # `IterableDataset.map` does not support `desc`
            map_kwargs['desc'] = f'Tokenizing {data_args.data_path} dataset'
        dataset = dataset.map(
            lambda ex: processing_class(ex[data_args.dataset_text_field]),
            **map_kwargs)

        # Pack or truncate
        if data_args.packing:
            if data_args.max_seq_length is None:
                raise ValueError(
                    "When packing is enabled, `max_seq_length` can't be `None`."
                )
            if isinstance(dataset, Dataset):
                # `IterableDataset.map` does not support `desc`
                map_kwargs['desc'] = f'Packing {data_args.data_path} dataset'
            dataset = dataset.select_columns('input_ids')
            dataset = dataset.map(
                pack_examples,
                batched=True,
                fn_kwargs={'seq_length': data_args.max_seq_length},
                **map_kwargs)
        elif data_args.max_seq_length is not None:
            dataset = dataset.map(
                lambda ex: {
                    key: ex[key][:data_args.max_seq_length]
                    for key in ['input_ids', 'attention_mask']
                },
                **map_kwargs,
            )
    return dataset


def prepare_dataset(
    dataset: Optional[Dataset] = None,
    processing_class: Optional[Union[PreTrainedTokenizer, BaseImageProcessor,
                                     FeatureExtractionMixin,
                                     ProcessorMixin]] = None,
    formatting_func: Optional[Callable] = None,
    data_args: DataArguments = None,
):
    if dataset is None:
        raise ValueError('The dataset should not be None')

    if data_args.skip_prepare_dataset:
        return dataset

    # If the dataset is already preprocessed (tokenized), return as-is. Only works if dataset is
    # a datasets.Dataset or datasets.IterableDataset -- not for torch Dataset
    column_names = (dataset.column_names if isinstance(
        dataset, (datasets.Dataset, datasets.IterableDataset)) else None)
    if column_names and 'input_ids' in column_names:
        if formatting_func is not None:
            warnings.warn(
                'You passed a dataset that is already processed (contains an `input_ids` field) together with a '
                'valid formatting function. Therefore `formatting_func` will be ignored. Either remove the '
                '`formatting_func` or pass a dataset that is not already processed.',
                UserWarning,
            )

        def formatting_func(x):
            return x['input_ids']

        if not data_args.packing:
            return dataset

    # check if torch dataset / dataloader and do nothing
    # see https://github.com/huggingface/trl/pull/1468 for why datasets.IterableDataset needs a separate check
    if isinstance(dataset,
                  (torch.utils.data.IterableDataset, torch.utils.data.Dataset,
                   ConstantLengthDataset)) and not isinstance(
                       dataset, datasets.IterableDataset):
        return dataset

    if not data_args.packing:
        return prepare_non_packed_dataloader(dataset, processing_class,
                                             formatting_func, data_args)

    else:
        return prepare_packed_dataloader(dataset, processing_class,
                                         formatting_func, data_args)


def prepare_non_packed_dataloader(
    dataset: Optional[Dataset] = None,
    processing_class: Optional[Union[PreTrainedTokenizer, BaseImageProcessor,
                                     FeatureExtractionMixin,
                                     ProcessorMixin]] = None,
    formatting_func: Optional[Callable] = None,
    data_args: DataArguments = None,
):
    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        outputs = processing_class(
            element[data_args.dataset_text_field]
            if formatting_func is None else formatting_func(element),
            add_special_tokens=data_args.add_special_tokens,
            truncation=True,
            padding=False,
            max_length=data_args.max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        if formatting_func is not None and not isinstance(
                formatting_func(element), list):
            raise ValueError(
                'The `formatting_func` should return a list of processed strings since it can lead to silent bugs.'
            )

        return {
            'input_ids': outputs['input_ids'],
            'attention_mask': outputs['attention_mask']
        }

    signature_columns = ['input_ids', 'labels', 'attention_mask']

    if dataset.column_names is not None:  # None for IterableDataset
        extra_columns = list(
            set(dataset.column_names) - set(signature_columns))
    else:
        extra_columns = []

    if not data_args.remove_unused_columns and len(extra_columns) > 0:
        warnings.warn(
            'You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with '
            'the default collator and yield to errors. If you want to inspect dataset other columns (in this '
            f'case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the '
            'default collator and create your own data collator in order to inspect the unused dataset columns.',
            UserWarning,
        )

    map_kwargs = {
        'batched':
        True,
        'remove_columns':
        dataset.column_names if data_args.remove_unused_columns else None,
        'batch_size':
        data_args.dataset_batch_size,
    }
    if isinstance(dataset, datasets.Dataset):
        map_kwargs[
            'num_proc'] = data_args.dataset_num_proc  # this arg is not available for IterableDataset
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    return tokenized_dataset


def prepare_packed_dataloader(
    dataset: Optional[Dataset] = None,
    processing_class: Optional[Union[PreTrainedTokenizer, BaseImageProcessor,
                                     FeatureExtractionMixin,
                                     ProcessorMixin]] = None,
    formatting_func: Optional[Callable] = None,
    data_args: DataArguments = None,
):
    if processing_class is None:
        raise ValueError(
            'You need to pass a processing_class with `SFTTrainer`.')

    constant_length_iterator = ConstantLengthDataset(
        processing_class,
        dataset,
        dataset_text_field=None
        if formatting_func is not None else data_args.dataset_text_field,
        formatting_func=formatting_func,
        seq_length=data_args.max_seq_length,
        infinite=False,
        num_of_sequences=data_args.num_of_sequences,
        chars_per_token=data_args.chars_per_token,
        eos_token_id=processing_class.eos_token_id,
        append_concat_token=data_args.append_concat_token,
        add_special_tokens=data_args.add_special_tokens,
    )

    if isinstance(dataset, datasets.IterableDataset):
        return constant_length_iterator

    def data_generator(constant_length_iterator):
        yield from constant_length_iterator

    try:
        packed_dataset = Dataset.from_generator(
            data_generator,
            gen_kwargs={'constant_length_iterator': constant_length_iterator})
    except (DatasetGenerationError, SchemaInferenceError) as exc:
        raise ValueError(
            'Error occurred while packing the dataset. '
            'Make sure that your dataset has enough samples to at least yield one packed sequence.'
        ) from exc
    return packed_dataset


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Attributes:
        PROMPT_DICT (dict): A dictionary containing prompts for the model to complete.
        IGNORE_INDEX (int): A value to replace tokens corresponding to the source in the labels tensor.

    Methods:
        __init__(self, data_path: str, tokenizer: PreTrainedTokenizer): Initializes a SupervisedDataset object.
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx) -> Dict[str, torch.Tensor]: Retrieves an example from the dataset at the specified index.
    """

    PROMPT_DICT = {
        'prompt_input':
        ('Below is an instruction that describes a task, paired with an input that provides further context. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
         ),
        'prompt_no_input':
        ('Below is an instruction that describes a task. '
         'Write a response that appropriately completes the request.\n\n'
         '### Instruction:\n{instruction}\n\n### Response:'),
    }
    IGNORE_INDEX = -100

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024):
        """Initializes a SupervisedDataset object.

        Args:
            data_path (str): The path to the training data file.
            tokenizer (PreTrainedTokenizer): The tokenizer object used to tokenize the input examples.
        """
        super(SupervisedDataset, self).__init__()
        logger.info(f'Loading dataset from {data_path}')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = load_dataset('json',
                                          data_files=data_path)['train']
        else:
            list_data_dict = load_dataset(data_path)['train']

        logger.info('Found %d rows', list_data_dict.num_rows)
        prompt_input, prompt_no_input = self.PROMPT_DICT[
            'prompt_input'], self.PROMPT_DICT['prompt_no_input']
        self.sources = [
            prompt_input.format_map(example) if example.get('input', '') != ''
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        self.targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        self.examples = [s + t for s, t in zip(self.sources, self.targets)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Retrieves an example from the dataset at the specified index.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the input_ids, labels, input_len, source_input_ids, and
            source_len tensors.
        """
        example_txt = self.examples[idx]
        # Tokenize the example and source text
        example_tokenized = self.tokenizer(
            example_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        source_txt = self.sources[idx]
        source_tokenized = self.tokenizer(
            source_txt,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
        )
        # Extract the input_ids tensor
        input_ids = torch.tensor(example_tokenized['input_ids'])
        # Create the labels tensor
        labels = input_ids.clone()
        labels[:len(source_tokenized['input_ids'])] = self.IGNORE_INDEX
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
