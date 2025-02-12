"""Part of the code was adopted from https://github.com/microsoft/Megatron-
DeepSpeed/blob/main/megatron/data/dataset_utils.py."""
import hashlib
import os
from itertools import chain
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from openr1.dataset.raw_datasets import (
    AlpacaChinese, AlpacaCoT, AlpacaDataset, AnthropicHHRLHF,
    BelleGroupDataset, CohereMiracljaqueries2212Dataset,
    CohereMiraclzhqueries2212Dataset, DahoasFullhhrlhfDataset,
    DahoasRmstaticDataset, DahoasSyntheticinstructgptjpairwiseDataset,
    DatabricksDolly15k, FudanMossDataset, Gpt4allPromptGeneration,
    GuanacoDataset, HelloSimpleAIHC3ChineseDataset, HuatuoMedDataset,
    InstructWildDataset, LaionOIG, LmqgQagjaquadDataset, LmqgQgjaquadDataset,
    MkqaChineseDataset, MkqaJapaneseDataset, MosaicmlDollyHHRLHF,
    OpenaiWebgptcomparisonsDataset, OpenAssistantOasst1, PromptDataset,
    PromptRawDataset, StackExchangeParied, StanfordnlpSHPDataset,
    Wangrui6ZhihuKOLDataset, YeungNLPFirefly,
    YitingxieRlhfrewarddatasetsDataset)

# Create a dictionary mapping dataset names to their corresponding Dataset classes
HuggingFaceDataClass: Dict[str, Type] = {
    'Dahoas/rm-static': DahoasRmstaticDataset,
    'Dahoas/full-hh-rlhf': DahoasFullhhrlhfDataset,
    'Dahoas/synthetic-instruct-gptj-pairwise':
    DahoasSyntheticinstructgptjpairwiseDataset,
    'yitingxie/rlhf-reward-datasets': YitingxieRlhfrewarddatasetsDataset,
    'openai/webgpt_comparisons': OpenaiWebgptcomparisonsDataset,
    'stanfordnlp/SHP': StanfordnlpSHPDataset,
    'wangrui6/Zhihu-KOL': Wangrui6ZhihuKOLDataset,
    'Cohere/miracl-zh-queries-22-12': CohereMiraclzhqueries2212Dataset,
    'Hello-SimpleAI/HC3-Chinese': HelloSimpleAIHC3ChineseDataset,
    'mkqa-Chinese': MkqaChineseDataset,
    'mkqa-Japanese': MkqaJapaneseDataset,
    'Cohere/miracl-ja-queries-22-12': CohereMiracljaqueries2212Dataset,
    'lmqg/qg_jaquad': LmqgQgjaquadDataset,
    'lmqg/qag_jaquad': LmqgQagjaquadDataset,
    'Anthropic/hh-rlhf': AnthropicHHRLHF,
    'databricks/databricks-dolly-15k': DatabricksDolly15k,
    'mosaicml/dolly_hhrlhf': MosaicmlDollyHHRLHF,
    'JosephusCheung/GuanacoDataset': GuanacoDataset,
    'YeungNLP/firefly-train-1.1M': YeungNLPFirefly,
    'OpenAssistant/oasst1': OpenAssistantOasst1,
    'tatsu-lab/alpaca': AlpacaDataset,
    'yahma/alpaca-cleaned': AlpacaDataset,
    'QingyiSi/Alpaca-CoT': AlpacaCoT,
    'fnlp/moss-002-sft-data': FudanMossDataset,
    'nomic-ai/gpt4all-j-prompt-generations': Gpt4allPromptGeneration,
    'lvwerra/stack-exchange-paired': StackExchangeParied,
    'laion/OIG': LaionOIG,
    'BelleGroup/train_1M_CN': BelleGroupDataset,
    'BelleGroup/train_0.5M_CN': BelleGroupDataset,
    'huatuo_med_data/llama_med': HuatuoMedDataset,
    'huatuo_med_data/liver_cancer': HuatuoMedDataset,
    'InstructionWild/instinwild_en': InstructWildDataset,
    'InstructionWild/instinwild_ch': InstructWildDataset,
    'alpaca_chinese/alpaca_data_zh_51k': AlpacaChinese,
    'alpaca_chinese/trans_chinese_alpaca_data': AlpacaChinese,
}


def get_raw_dataset(dataset_name: Optional[str] = None,
                    data_dir: Optional[str] = None,
                    test_data_ratio: float = 0.1,
                    seed: Optional[int] = None):
    """Given a dataset_name, returns an instance of the corresponding Dataset
    class, initialized with the given test_data_ratio and seed arguments.

    Args:
        dataset_name (str, optional): Name of the dataset to return.
                                      Defaults to None.
        test_data_ratio (float, optional): Ratio of test data to include in the returned dataset.
                                           Defaults to 0.1.
        seed (int, optional): Seed used for generating random numbers.
                              Defaults to None.

    Returns:
        An instance of the corresponding Dataset class with the provided parameters.

    Raises:
        RuntimeError: If no Dataset class is defined for the given dataset_name.
    """
    if dataset_name in HuggingFaceDataClass:
        # Create an instance of the corresponding Dataset class with the provided parameters
        return HuggingFaceDataClass[dataset_name](
            dataset_name=dataset_name,
            data_dir=data_dir,
            test_data_ratio=test_data_ratio,
            seed=seed,
        )
    else:
        raise RuntimeError(
            f'We do not have define dataset {dataset_name}, but you can add it by yourself in raw_dataset.py.'
        )


def data_preprocess(
        current_dataset: Dataset,
        raw_dataset: PromptRawDataset,
        train_phase: int = 1,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_seq_len: int = 512,
        end_of_conversation_token: Optional[str] = None) -> PromptDataset:
    """Create different splits of a dataset based on the training phase.

    Args:
        current_dataset (Dataset): The current state of the dataset.
        raw_dataset (PromptRawDataset): The raw version of the dataset.
        train_phase (int, optional): The phase of training the model is in. Defaults to 1.
        tokenizer (Optional[PreTrainedTokenizer], optional): The tokenizer to use for tokenizing the text data. \
          Defaults to None.
        max_seq_len (int, optional): The maximum length for each sequence. Defaults to 512.
        end_of_conversation_token (Optional[str], optional): A special end-of-conversation token that will be added \
            to the end of each response if provided. Defaults to None.

    Returns:
        PromptDataset: An instance of the PromptDataset class containing the prompt_dataset, chosen_dataset, and \
            reject_dataset, along with other relevant information such as the tokenizer, max_length, and train_phase.
    """
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []

    for i, raw_sample in enumerate(current_dataset):
        if train_phase == 1:
            # Get the chosen response
            chosen_sentence = raw_dataset.get_prompt_and_chosen(raw_sample)
            # Add end_of_conversation_token to the chosen response if provided
            if chosen_sentence is not None and end_of_conversation_token is not None:
                chosen_sentence += end_of_conversation_token
                chosen_dataset.append(chosen_sentence)

        elif train_phase == 2:
            # Get the chosen and rejected responses
            chosen_sentence = raw_dataset.get_prompt_and_chosen(raw_sample)
            reject_sentence = raw_dataset.get_prompt_and_rejected(raw_sample)
            # Add end_of_conversation_token to the chosen and rejected responses if provided
            if chosen_sentence is not None and reject_sentence is not None and end_of_conversation_token is not None:
                chosen_sentence += end_of_conversation_token
                reject_sentence += end_of_conversation_token
                chosen_dataset.append(chosen_sentence)
                reject_dataset.append(reject_sentence)

        elif train_phase == 3:
            # Get the prompt only
            prompt = raw_dataset.get_prompt(raw_sample)
            if prompt is not None:
                prompt_dataset.append(prompt)

    return PromptDataset(prompt_dataset=prompt_dataset,
                         chosen_dataset=chosen_dataset,
                         reject_dataset=reject_dataset,
                         tokenizer=tokenizer,
                         max_length=max_seq_len,
                         train_phase=train_phase)


def create_dataset(
    dataset_name: str,
    data_dir: Optional[str] = None,
    train_phase: Optional[int] = 1,
    test_data_ratio: float = 0.1,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_seq_len: int = 512,
    end_of_conversation_token: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple:
    """A function that creates a training and evaluation dataset by splitting a
    raw dataset.

    Args:
    - dataset_name (str): The name of the dataset to load.
    - train_phase (int, optional): An integer indicating the current phase of training.
                                    Used for creating subsets of data for different phases of training.
    - test_data_ratio (float, default=0.1): A float indicating the ratio of test data to total data.
    - tokenizer (PreTrainedTokenizer, optional): An object used for tokenizing text data.
    - max_seq_len (int, default=512): An integer indicating the maximum length of token sequences.
    - end_of_conversation_token (str, optional): A string token that marks the end of a conversation.
    - seed (int, optional): An integer used for setting random seed value for reproducibility purposes.

    Returns:
    - A tuple containing two datasets: train and eval datasets.
    """

    # Load the raw dataset using the given name, test_data_ratio and seed
    raw_dataset = get_raw_dataset(dataset_name,
                                  data_dir=data_dir,
                                  test_data_ratio=test_data_ratio,
                                  seed=seed)
    # Get the training dataset from the raw dataset
    train_dataset = raw_dataset.get_train_data()

    # Create a split of the training dataset using create_dataset_split function
    train_dataset = data_preprocess(
        current_dataset=train_dataset,
        raw_dataset=raw_dataset,
        train_phase=train_phase,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        end_of_conversation_token=end_of_conversation_token)

    # Get the evaluation dataset from the raw dataset
    eval_dataset = raw_dataset.get_eval_data()

    # Create a split of the evaluation dataset using create_dataset_split function
    eval_dataset = data_preprocess(
        current_dataset=eval_dataset,
        raw_dataset=raw_dataset,
        train_phase=train_phase,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        end_of_conversation_token=end_of_conversation_token)

    return train_dataset, eval_dataset


def create_prompt_dataset(
    dataset_names: list = None,
    data_dir: Optional[str] = None,
    train_phase: int = 1,
    test_data_ratio: float = 0.1,
    tokenizer: PreTrainedTokenizer = None,
    max_seq_len: int = 512,
    end_of_conversation_token: str = '<|endoftext|>',
    output_path: str = None,
    seed: int = None,
):
    """Creates the prompt dataset."""
    os.makedirs(output_path, exist_ok=True)
    fname = '_'.join(dataset_names)
    tokenizer_name = tokenizer.init_kwargs['name_or_path'].replace('/', '_')
    fname = f'{fname}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}'

    fname = '_'.join(fname.split('/'))
    fname = hashlib.sha256(fname.encode()).hexdigest()
    # hash the file name to avoid too long file name
    train_fname = f'{output_path}/traindata_{fname}.pt'
    eval_fname = f'{output_path}/evaldata_{fname}.pt'

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    # Skip creating cache if we found it on all the nodes.
    if cache_found:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        train_datasets = []
        eval_datasets = []
        train_size = 0
        eval_size = 0
        for d_name in dataset_names:
            train_dataset, eval_dataset = create_dataset(
                dataset_name=d_name,
                data_dir=data_dir,
                train_phase=train_phase,
                test_data_ratio=test_data_ratio,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                end_of_conversation_token=end_of_conversation_token,
                seed=seed,
            )
            print(
                f'Ceate dataset, {d_name}, train size: {len(train_dataset)}, eval size: {len(eval_dataset)}'
            )
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_size += len(train_dataset)
            eval_size += len(eval_dataset)
        train_dataset = ConcatDataset(train_datasets)
        eval_dataset = ConcatDataset(eval_datasets)
        print(
            f'Concate dataset: {train_datasets}, train size: {len(train_dataset)}, eval size: {len(eval_dataset)}'
        )
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    return train_dataset, eval_dataset


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch['input_ids'] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch['attention_mask'] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        # make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch['prompt'] = F.pad(prompt,
                                    pad=(pad_length, 0),
                                    mode='constant',
                                    value=pad_token_id)
            batch['prompt_att_mask'] = F.pad(prompt_mask,
                                             pad=(pad_length, 0),
                                             mode='constant',
                                             value=0)
        else:
            batch['prompt'] = prompt
            batch['prompt_att_mask'] = prompt_mask
        batch['prompt'] = batch['prompt'].flip(1)
        batch['prompt_att_mask'] = batch['prompt_att_mask'].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f'Grouping texts in chunks of {block_size}',
    )

    train_dataset = lm_datasets['train']

    return train_dataset
