import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from datasets import load_dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, LlamaTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, Trainer, TrainingArguments)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={'help': 'Path to the training data.'})


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


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj'])
    lora_weight_path: str = ''
    bias: str = 'none'


def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.cpu().clone().detach()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(state_dict, bias):
    if bias == 'none':
        to_return = {
            k: state_dict[k].cpu().clone().detach()
            for k in state_dict if 'lora_' in k
        }
    elif bias == 'all':
        to_return = {
            k: state_dict[k]
            for k in state_dict if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict,
                                         tokenizer: PreTrainedTokenizer,
                                         model: PreTrainedModel):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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
        logging.warning(f'Loading dataset from {data_path}')
        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = load_dataset('json',
                                          data_files=data_path)['train']
        else:
            list_data_dict = load_dataset(data_path)['train']

        logging.warning('Found %d rows', list_data_dict.num_rows)
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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = pad_sequence(input_ids,
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels,
                              batch_first=True,
                              padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train(load_in_8bit=False) -> None:
    """Trains a language model using Hugging Face's Transformers library.

    Args:
        model_args (ModelArguments): The arguments for the model configuration.
        data_args (DataArguments): The arguments for the data configuration.
        training_args (TrainingArguments): The arguments for the training configuration.
        lora_args (LoraArguments): The arguments for low-rank factorization.

    Returns:
        None
    """
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses(
    )
    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    if ddp:
        device_map = {'': int(os.environ.get('LOCAL_RANK') or 0)}

    # Load the pre-trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
    )

    # Set up LORA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type='CAUSAL_LM',
    )

    # Warn about potential issue with gradient checkpointing and LORA
    if training_args.gradient_checkpointing:
        logging.warning(
            'Gradient checkpointing with LORA makes requires_grad '
            'incorrect and needs a monkey patch in Trainer or the '
            "wrapped model's forward. Ref: "
            'https://github.com/lm-sys/FastChat/pull/138#issuecomment-1509172198'
        )

    # Load the tokenizer
    if model.config.model_type == 'llama':
        # Due to the name of Transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side='right',
            use_fast=True,
        )

    # Resize the tokenizer's vocabulary size to accommodate additional special tokens, if necessary
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    special_tokens_dict['additional_special_tokens'] = [
        '### Instruction:',
        '### Response:\n',
        '### End',
    ]

    if len(special_tokens_dict) > 0:
        smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                             model)

    # Prepare the model for int8 training and get the PEFT model
    if load_in_8bit:
        model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)

    # Print the percentage of trainable parameters if using DeepSpeed and running on local rank 0
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    max_length = None
    for length_setting in [
            'n_positions', 'max_position_embeddings', 'seq_length'
    ]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logging.warning(f'Found max lenth: {max_length}')
            break
    if not max_length:
        max_length = 1024
        logging.warning(f'Using default max length: {max_length}')

    # Create a supervised dataset and Trainer, then train the model
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    if training_args.resume_from_checkpoint and list(
            pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    # Save the trained model
    trainer.save_model(training_args.output_dir)

    # Save states. Weights might be a placeholder in zero3 and need a gather
    state_dict = get_peft_state_maybe_zero_3(model.state_dict(),
                                             lora_args.bias)
    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)


if __name__ == '__main__':
    train(load_in_8bit=True)
