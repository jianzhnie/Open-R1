import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def get_dataset_split_index(data_size, test_data_ratio, seed):

    index_list = list(range(data_size))
    triain_index, test_index = train_test_split(index_list,
                                                test_size=test_data_ratio,
                                                random_state=seed)
    return triain_index, test_index


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


class PromptDataset(Dataset):
    """A PyTorch dataset class that prepares prompt sentences and their
    corresponding chosen/rejected sentences for training.

    Args:
        prompt_dataset (List): A list of prompt sentences.
        chosen_dataset (List): A list of chosen sentences.
        reject_dataset (List): A list of rejected sentences.
        tokenizer (PreTrainedTokenizer): A pre-trained tokenizer from the Hugging Face transformers library.
        max_length (int): Maximum length of encoded sequences. Default is 512.
        train_phase (int): Phase of training data to prepare. Can be 1, 2 or 3. Default is 1.

    Returns:
        Dictionary or tuple of tensors depending on the value of train_phase.

    Examples:
        >>> prompt_dataset = ['What is your favorite color?', 'Do you like pizza?']
        >>> chosen_dataset = ['My favorite color is blue.', 'I love pizza!']
        >>> reject_dataset = ['I don\'t have a favorite color.', 'Pizza is not my thing.']
        >>> tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        >>> dataset = PromptDataset(prompt_dataset, chosen_dataset, reject_dataset, tokenizer, max_length=256)
    """

    def __init__(self,
                 prompt_dataset: List[str],
                 chosen_dataset: List[str],
                 reject_dataset: List[str],
                 tokenizer: PreTrainedTokenizer = None,
                 max_length: int = 512,
                 train_phase: int = 1) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length
        self.train_phase = train_phase

    def __len__(self) -> int:
        """Returns the length of chosen_dataset, or prompt_dataset if
        train_phase is 3.

        Returns:
            int: Length of dataset.
        """
        if self.train_phase == 3:
            return len(self.prompt_dataset)
        else:
            return len(self.chosen_dataset)

    def __getitem__(self, idx: int) -> dict or Tuple[torch.Tensor]:
        """Returns a dictionary or tuple of tensors depending on the value of
        train_phase.

        Args:
            idx (int): Index of dataset item to retrieve.

        Returns:
            Dictionary or tuple of tensors depending on the value of train_phase.
        """
        if self.train_phase == 1:
            raw_input = self.chosen_dataset[idx]
            encoding_input = self.tokenizer(raw_input,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length')
            # set labels equal to input_ids to enable computing loss later
            encoding_input['labels'] = encoding_input['input_ids']
            encoding_input = {
                key: torch.tensor(val)
                for key, val in encoding_input.items()
            }
            return {
                'input_ids': encoding_input['input_ids'],
                'attention_mask': encoding_input['attention_mask'],
                'labels': encoding_input['labels']
            }

        elif self.train_phase == 2:
            chosen_sentence = self.chosen_dataset[idx]
            reject_sentence = self.reject_dataset[idx]
            chosen_input = self.tokenizer(chosen_sentence,
                                          truncation=True,
                                          max_length=self.max_length,
                                          padding='max_length')
            reject_input = self.tokenizer(reject_sentence,
                                          truncation=True,
                                          max_length=self.max_length,
                                          padding='max_length')

            chosen_input = {
                key: torch.tensor(val)
                for key, val in chosen_input.items()
            }
            reject_input = {
                key: torch.tensor(val)
                for key, val in reject_input.items()
            }

            return (chosen_input['input_ids'], chosen_input['attention_mask'],
                    chosen_input['labels'], reject_input['input_ids'],
                    reject_input['attention_mask'])

        elif self.train_phase == 3:
            raw_input = self.prompt_dataset[idx]
            encoding_input = self.tokenizer(raw_input,
                                            truncation=True,
                                            max_length=self.max_length,
                                            padding='max_length')
            encoding_input = {
                key: torch.tensor(val)
                for key, val in encoding_input.items()
            }

            return (encoding_input['input_ids'],
                    encoding_input['attention_mask'], self.pad_token_id)


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.


class PromptRawDataset(object):
    """A class to handle raw text data for prompt-based dialogue systems.

    Attributes:
        dataset_name (str): The name of the dataset to load.
        dataset_name_clean (str): The cleaned version of the dataset name.
        test_data_ratio (float): The ratio of data to split as test data.
        seed (int): The random seed to use for data splitting.
        raw_datasets: The raw dataset loaded using `load_dataset()` function.

    Methods:
        get_train_data(): Returns the training data.
        get_eval_data(): Returns the evaluation data.
        get_prompt(sample: Dict[str, Any]) -> str: Returns the formatted prompt for a given sample.
        get_chosen(sample: Dict[str, Any]) -> Optional[str]: Returns the chosen response for a given sample, \
        if available.
        get_rejected(sample: Dict[str, Any]) -> Optional[str]: Returns the rejected response for a given sample, \
            if available.
        get_prompt_and_chosen(sample: Dict[str, Any]) -> Tuple[str, Optional[str]]: Returns the formatted prompt \
            and chosen response for a given sample.
        get_prompt_and_rejected(sample: Dict[str, Any]) -> Tuple[str, Optional[str]]: Returns the formatted prompt \
            and rejected response for a given sample, if available.
    """

    def __init__(self,
                 dataset_name: str,
                 data_dir: str = None,
                 num_proc: Optional[int] = 8,
                 test_data_ratio: Optional[float] = 0.1,
                 seed: Optional[int] = None):
        """Initializes the PromptRawDataset object.

        Args:
            dataset_name (str): The name of the dataset to load.
            num_proc (int, optional): The number of processes to use for parallel loading. Defaults to 8.
            test_data_ratio (float, optional): The ratio of data to split as test data. Defaults to 0.1.
            seed (int, optional): The random seed to use for data splitting. Defaults to None.
        """
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name.replace('/', '_')
        self.num_proc = num_proc
        self.test_data_ratio = test_data_ratio
        self.seed = seed

        if data_dir is not None:
            path = os.path.join(data_dir, dataset_name)
            if os.path.exists(path) and os.listdir(path):
                print(f'load {dataset_name}  from {path}')
                self.raw_datasets = load_dataset(path, num_proc=num_proc)
            else:
                self.raw_datasets = load_dataset(dataset_name,
                                                 num_proc=num_proc)
        else:
            self.raw_datasets = load_dataset(dataset_name, num_proc=num_proc)

    def get_train_data(self):
        """Returns the training data.

        Returns:
            datasets.Dataset: The training data.
        """
        return self.raw_datasets['train']

    def get_eval_data(self):
        """Returns the evaluation data.

        Returns:
            datasets.Dataset: The evaluation data.
        """
        return self.raw_datasets['validation']

    def get_prompt(self, sample: Dict[str, Any]) -> str:
        """Returns the formatted prompt for a given sample.

        Args:
            sample (Dict[str, Any]): The sample to generate a prompt for.

        Returns:
            str: The formatted prompt.
        """
        return f"Human: {sample['prompt']} Assistant:"

    def get_chosen(self, sample: Dict[str, Any]) -> Optional[str]:
        """Returns the chosen response for a given sample, if available.

        Args:
            sample (Dict[str, Any]): The sample to retrieve the chosen response from.

        Returns:
            str or None: The chosen response, or None if not available.
        """
        if 'chosen_response' in sample:
            return f" {sample['chosen_response']}"
        else:
            return None

    def get_rejected(self, sample: Dict[str, Any]) -> Optional[str]:
        """Returns the rejected response for a given sample, if available.

        Args:
            sample (Dict[str, Any]): The sample to retrieve the rejected response from.

        Returns:
            str or None: The rejected response, or None if not available.
        """
        if 'rejected_responses' in sample and len(
                sample['rejected_responses']) > 0:
            return f" {random.choice(sample['rejected_responses'])}"
        else:
            return None

    def get_prompt_and_chosen(
            self, sample: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Returns the formatted prompt and chosen response for a given sample.

        Args:
            sample (Dict[str, Any]): The sample to generate the prompt and retrieve the chosen response from.

        Returns:
            Tuple[str, Optional[str]]: The formatted prompt and chosen response, or None if not available.
        """
        prompt = self.get_prompt(sample)
        chosen = self.get_chosen(sample)
        return prompt + '' + chosen

    def get_prompt_and_rejected(
            self, sample: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Returns the formatted prompt and rejected response for a given
        sample.

        Args:
            sample (Dict[str, Any]): The sample to generate the prompt and retrieve the rejected response from.

        Returns:
            Tuple[str, Optional[str]]: The formatted prompt and rejected response, or None if not available.
        """
        prompt = self.get_prompt(sample)
        rejected = self.get_rejected(sample)
        return prompt + '' + rejected


# checked
class StackExchangeParied(PromptRawDataset):
    """https://huggingface.co/datasets/lvwerra/stack-exchange-paired."""

    def __init__(
        self,
        dataset_name='lvwerra/stack-exchange-paired',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['response_j']

    def get_rejected(self, sample):
        return ' ' + sample['response_k']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
            'response_j']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
            'response_k']


# TODO
class AnthropicHHRLHF(PromptRawDataset):
    """https://huggingface.co/datasets/Anthropic/hh-rlhf."""

    def __init__(
        self,
        dataset_name='Anthropic/hh-rlhf',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['chosen'].split('Assistant:')[0] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['chosen'].split('Assistant:')[1]

    def get_rejected(self, sample):
        return ' ' + sample['rejected'].split('Assistant:')[1]

    def get_prompt_and_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['rejected']


# TODO
# [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
class DatabricksDolly15k(PromptRawDataset):
    """https://huggingface.co/datasets/databricks/databricks-dolly-15k."""

    def __init__(
        self,
        dataset_name='databricks/databricks-dolly-15k',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        PROMPT_DICT = {
            'prompt_input':
            ('Below is an instruction that describes a task, paired with an input that provides further context. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:'
             ),
            'prompt_no_input':
            ('Below is an instruction that describes a task. '
             'Write a response that appropriately completes the request.\n\n'
             '### Instruction:\n{instruction}\n\n### Response:'),
        }

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('context', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['response']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('context', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['response']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
class MosaicmlDollyHHRLHF(PromptRawDataset):
    """https://huggingface.co/datasets/mosaicml/dolly_hhrlhf.

    This dataset is a combination of Databrick's dolly-15k dataset and a filtered subset of Anthropic's HH-RLHF.
    It also includes a test split, which was missing in the original dolly set.
    That test set is composed of 200 randomly selected samples from dolly + 4,929 of the test set samples from HH-RLHF \
        which made it through the filtering process.
    The train set contains 59,310 samples; 15,014 - 200 = 14,814 from Dolly, and the remaining 44,496 from HH-RLHF.

    It is slightly larger than Alpaca, and in our experience of slightly higher quality, \
        but is usable for commercial purposes so long as you follow the terms of the license.
    """

    def __init__(
        self,
        dataset_name='mosaicml/dolly_hhrlhf',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return ' Human: ' + sample['prompt']

    def get_chosen(self, sample):
        return sample['response']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['prompt'] + sample['response']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


class GuanacoDataset(PromptRawDataset):
    """https://huggingface.co/datasets/JosephusCheung/GuanacoDataset.

    Guanaco 模型的数据集旨在增强多语言能力并解决各种语言任务。它以 Alpaca 模型的 175 个任务为基础，、
    提供了用不同语言重写的种子任务，并添加了专门为英语语法分析、自然语言理解、跨语言自我意识和显式内容识别设计的新任务。
    该数据集总共包含 534,530 个条目，以 6000 美元的低成本生成。

    The dataset for the Guanaco model is designed to enhance the multilingual capabilities and address various \
    linguistic tasks. It builds upon the 175 tasks from the Alpaca model by providing rewrites of seed tasks \
    in different languages and adding new tasks specifically designed for English grammar analysis, \
    natural language understanding, cross-lingual self-awareness, and explicit content recognition. \
    The dataset comprises a total of 534,530 entries, generated at a low cost of $6K.

    Free chat dialogues without System input: 32,880 entries (recent update) - in English zh-Hans \
        zh-Hant-TW Japanese Deutsch

    To test 0-shot tasks of Japanese & Deutsch on original 175 tasks with finetuning on chat only.

    Chat dialogues with System input: 16,087 entries (recent update) - in English zh-Hans zh-Hant-TW zh-Hant-HK
    """

    def __init__(
        self,
        dataset_name='JosephusCheung/GuanacoDataset',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


class YeungNLPFirefly(PromptRawDataset):
    """https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M.

    本数据应用于项目：Firefly（流萤）: 中文对话式大语言模型 ，训练后得到的模型firefly-1b4

    如果您觉得此数据集对您有帮助，请like此数据集并在Github项目中star我们。

    我们收集了23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万 。

    每条数据的格式如下，包含任务类型、输入、目标输出：

    {
    "kind": "ClassicalChinese",
    "input": "将下面句子翻译成现代文：\n石中央又生一树，高百余尺，条干偃阴为五色，翠叶如盘，花径尺余，色深碧，蕊深红，异香成烟，著物霏霏。",
    "target": "大石的中央长着一棵树，一百多尺高，枝干是彩色的，树叶有盘子那样大，花的直径有一尺宽，花瓣深蓝色，花中飘出奇异的香气笼罩着周围，如烟似雾。"
    }
    """

    def __init__(
        self,
        dataset_name='YeungNLP/firefly-train-1.1M',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return ' Human: ' + sample['input']

    def get_chosen(self, sample):
        return ' ' + sample['target']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['input'] + sample['target']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
# [InstructWild Data](https://github.com/XueFuzhao/InstructionWild/tree/main/data)
class InstructWildDataset(object):
    """https://github.com/XueFuzhao/InstructionWild/tree/main/data."""

    def __init__(
        self,
        dataset_name='InstructionWild/instinwild_ch',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        assert data_dir is not None, 'data_dir must be specified.'
        path = os.path.join(data_dir, dataset_name + '.json')
        assert os.path.exists(path), f'{path} does not exist.'
        self.raw_datasets = load_dataset('json',
                                         data_files=path,
                                         num_proc=num_proc)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
class HuatuoMedDataset(object):
    """https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese/tree/main/data."""

    def __init__(
        self,
        dataset_name='huatuo_med_data/llama_data',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        assert data_dir is not None, 'data_dir must be specified.'
        path = os.path.join(data_dir, dataset_name + '.json')
        assert os.path.exists(path), f'{path} does not exist.'
        self.raw_datasets = load_dataset('json',
                                         data_files=path,
                                         num_proc=num_proc)
        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
# [laion/OIG](https://huggingface.co/datasets/laion/OIG)
class LaionOIG(PromptRawDataset):
    """https://huggingface.co/datasets/laion/OIG."""

    def __init__(
        self,
        dataset_name='laion/OIG',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['text']

    def get_chosen(self, sample):
        return sample['text']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return sample['text']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
# [OpenAssistant/oasst1](https://github.com/LAION-AI/Open-Assistant/blob/main/docs/docs/data/datasets.md)
class OpenAssistantOasst1(PromptRawDataset):
    """https://huggingface.co/datasets/OpenAssistant/oasst1."""

    def __init__(
        self,
        dataset_name='OpenAssistant/oasst1',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return sample['text'].split('<bot>:')[0] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['text'].split('<bot>:')[1]

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return sample['text']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
# [1.5M中文数据集](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)
class BelleGroupDataset(PromptRawDataset):
    """https://huggingface.co/datasets/BelleGroup/train_1M_CN.

    - 内容
        包含约200万条由BELLE项目生成的中文指令数据。

    - 样例
        {
        "instruction": "给定一个文字输入，将其中的所有数字加1。\n“明天的会议在9点开始，记得准时到达。”\n",
        "input": "",
        "output": "“明天的会议在10点开始，记得准时到达。”"
        }

    - 字段：
        instruction: 指令
        input: 输入（本数据集均为空）
        output: 输出
    """

    def __init__(
        self,
        dataset_name='BelleGroup/train_1M_CN',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
# [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
class AlpacaDataset(PromptRawDataset):
    """https://huggingface.co/datasets/tatsu-lab/alpaca."""

    def __init__(
        self,
        dataset_name='tatsu-lab/alpaca',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


class AlpacaCoT(PromptRawDataset):
    """https://huggingface.co/datasets/Dahoas/rm-static."""

    def __init__(
        self,
        dataset_name='QingyiSi/Alpaca-CoT',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)
        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


class AlpacaChinese(object):
    """https://github.com/LC1332/Luotuo-Chinese-LLM/tree/main/data
    https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/data."""

    def __init__(
        self,
        dataset_name='alpaca_chinese/trans_chinese_alpaca_data',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        assert data_dir is not None, 'data_dir must be specified.'
        path = os.path.join(data_dir, dataset_name + '.json')
        assert os.path.exists(path), f'{path} does not exist.'
        self.raw_datasets = load_dataset('json',
                                         data_files=path,
                                         num_proc=num_proc)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

        self.prompt_input, self.prompt_no_input = PROMPT_DICT[
            'prompt_input'], PROMPT_DICT['prompt_no_input']

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        return ' Human: ' + instruct + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['output']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample.get('input', '') != '':
            instruct = self.prompt_input.format_map(sample)
        else:
            instruct = self.prompt_no_input.format_map(sample)
        target = sample['output']
        if instruct is None or target is None:
            return None
        return ' Human: ' + instruct + ' Assistant: ' + target

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# TODO
class FudanMossDataset(PromptRawDataset):
    """https://huggingface.co/datasets/fnlp/moss-002-sft-data."""

    def __init__(
        self,
        dataset_name='fnlp/moss-002-sft-data',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ) -> None:

        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):

        return sample['plain_text']

    def get_chosen(self, sample):
        return sample['plain_text']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return sample['plain_text']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# English dataset
class Gpt4allPromptGeneration(PromptRawDataset):
    """https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-
    generations."""

    def __init__(
        self,
        dataset_name='nomic-ai/gpt4all-j-prompt-generations',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['response']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):
    """https://huggingface.co/datasets/Dahoas/rm-static."""

    def __init__(
        self,
        dataset_name='Dahoas/rm-static',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):
    """https://huggingface.co/datasets/Dahoas/full-hh-rlhf."""

    def __init__(
        self,
        dataset_name='Dahoas/full-hh-rlhf',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):
    """https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-
    pairwise."""

    def __init__(
        self,
        dataset_name='Dahoas/synthetic-instruct-gptj-pairwise',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['chosen']

    def get_rejected(self, sample):
        return ' ' + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant: ' + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['prompt'] + ' Assistant: ' + sample[
            'rejected']


# English
# todo: check
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):
    """https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets."""

    def __init__(
        self,
        dataset_name='yitingxie/rlhf-reward-datasets',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return sample['prompt'] + 'Assistant:'

    def get_chosen(self, sample):
        return sample['chosen'].split('Assistant:')[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split('Assistant:')[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    """https://huggingface.co/datasets/openai/webgpt_comparisons."""

    def __init__(
        self,
        dataset_name='openai/webgpt_comparisons',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        return ' Human: ' + sample['question']['full_text'] + ' Assistant:'

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' ' + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' ' + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' Human: ' + sample['question'][
            'full_text'] + ' Assistant: ' + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r' [\(\[].*?[\)\]]', '', response)
        response = re.sub(r'[\(\[].*?[\)\]]', '', response)
        return ' Human: ' + sample['question'][
            'full_text'] + ' Assistant: ' + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):
    """https://huggingface.co/datasets/stanfordnlp/SHP."""

    def __init__(
        self,
        dataset_name='stanfordnlp/SHP',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['history'] + ' Assistant:'

    def get_chosen(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_A']
        else:
            response = sample['human_ref_B']
        return ' ' + response

    def get_rejected(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_B']
        else:
            response = sample['human_ref_A']
        return ' ' + response

    def get_prompt_and_chosen(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_A']
        else:
            response = sample['human_ref_B']
        return ' Human: ' + sample['history'] + ' Assistant: ' + response

    def get_prompt_and_rejected(self, sample):
        if int(sample['labels']) == 1:
            response = sample['human_ref_B']
        else:
            response = sample['human_ref_A']
        return ' Human: ' + sample['history'] + ' Assistant: ' + response


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):
    """https://huggingface.co/datasets/wangrui6/Zhihu-KOL."""

    def __init__(
        self,
        dataset_name='wangrui6/Zhihu-KOL',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return ' Human: ' + sample['INSTRUCTION'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return ' ' + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return ' Human: ' + sample[
                'INSTRUCTION'] + ' Assistant: ' + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):
    """https://huggingface.co/datasets/Cohere/miracl-zh-queries-22-12."""

    def __init__(
        self,
        dataset_name='Cohere/miracl-zh-queries-22-12',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['dev']

    def get_prompt(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return ' ' + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):
    """https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese."""

    def __init__(
        self,
        dataset_name='Hello-SimpleAI/HC3-Chinese',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.datasets = self.raw_datasets['train']
        self.raw_datasets = self.datasets.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return ' Human: ' + sample['question'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return ' ' + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):
    """https://huggingface.co/datasets/mkqa."""

    def __init__(
        self,
        dataset_name='mkqa-Chinese',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return ' Human: ' + sample['queries']['zh_cn'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return ' ' + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return ' Human: ' + sample['queries'][
                'zh_cn'] + ' Assistant: ' + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(
        self,
        dataset_name='mkqa-Japanese',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

        self.dataset = self.raw_datasets['train']
        self.raw_datasets = self.dataset.train_test_split(
            test_size=test_data_ratio, seed=seed)

    def get_train_data(self):

        return self.raw_datasets['train']

    def get_eval_data(self):

        return self.raw_datasets['test']

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return ' Human: ' + sample['queries']['ja'] + ' Assistant:'
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return ' ' + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return ' Human: ' + sample['queries'][
                'ja'] + ' Assistant: ' + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(
        self,
        dataset_name='Cohere/miracl-ja-queries-22-12',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['dev']

    def get_prompt(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return ' ' + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return ' Human: ' + sample['query'] + ' Assistant: ' + sample[
            'negative_passages'][0]['text']


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):
    """https://huggingface.co/datasets/lmqg/qg_jaquad."""

    def __init__(
        self,
        dataset_name='lmqg/qg_jaquad',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['sentence']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['question'] + ' Assistant: ' + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(
        self,
        dataset_name='lmqg/qag_jaquad',
        data_dir: str = None,
        num_proc: int = 8,
        test_data_ratio: float = 0.1,
        seed=None,
    ):
        super().__init__(dataset_name, data_dir, num_proc, test_data_ratio,
                         seed)

    def get_train_data(self):
        return self.raw_datasets['train']

    def get_eval_data(self):
        return self.raw_datasets['validation']

    def get_prompt(self, sample):
        return ' Human: ' + sample['questions'][0] + ' Assistant:'

    def get_chosen(self, sample):
        return ' ' + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return ' Human: ' + sample['questions'][0] + ' Assistant: ' + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f'Warning: dataset {self.dataset_name} does not include rejected response.'
        )
        return None
