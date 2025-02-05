from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import (AutoModel, AutoTokenizer, BloomModel, GPT2Model,
                          OPTModel)
from transformers.modeling_outputs import ModelOutput

from .pairwise_loss import PairWiseLoss


@dataclass
class RewardModelOutput(ModelOutput):
    """A class representing the output of a reward-based machine learning
    model.

    Attributes:
        loss (`Optional[torch.FloatTensor]`, optional): The classification or regression loss of the model.
        rewards_chosen (`torch.FloatTensor`): The rewards for the chosen actions.
        rewards_rejected (`torch.FloatTensor`): The rewards for the rejected actions.
    """

    # Define class attributes with type annotations and default values
    loss: Optional[torch.FloatTensor] = None
    rewards_chosen: torch.FloatTensor = None
    rewards_rejected: torch.FloatTensor = None


class Pooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled = self.dense(first_token_tensor)
        pooled = self.activation(pooled)
        return pooled


class MeanPooler(nn.Module):
    """Applies a mean pooling on the hidden states of the last layer of the
    transformer model."""

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """Applies mean pooling on the hidden states of the last layer of the
        transformer model."""
        # Calculate the mean of the hidden states of the last layer
        pooled = hidden_states.mean(dim=1)
        # Apply a linear layer followed by a tanh activation function
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)
        return pooled


class PairedRewardModel(nn.Module):
    """GPT Reward model.

    Args:
        model (str): Model name: 'opt', 'gpt2' or 'bloom'
        pretrained (str): Pretrained model name or path.
    """

    def __init__(self, pretrained: str = 'openai-gpt'):
        super().__init__()

        # Instantiate model based on input string
        if 'opt' in pretrained:
            self.model = OPTModel.from_pretrained(pretrained)
        elif 'gpt2' in pretrained:
            self.model = GPT2Model.from_pretrained(pretrained)
        elif 'bloom' in pretrained:
            self.model = BloomModel.from_pretrained(pretrained)
        else:
            # If the model string is invalid, raise an error
            raise ValueError(
                "Invalid model name. Choose 'opt', 'gpt2' or 'bloom'.")

        # Get the model's config and create a value head
        self.config = self.model.config

        n_embed = self.config.hidden_size if hasattr(
            self.config, 'hidden_size') else self.config.n_embd
        if 'opt' in pretrained:
            n_embed = self.config.word_embed_proj_dim

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        # Set EOS token and padding token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.eos_token_id = 2
            # add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.PAD_ID = self.tokenizer.pad_token_id
        self.value_head = nn.Linear(n_embed, 1, bias=False)
        self.loss_fn = PairWiseLoss()

    def forward(
        self,
        chosen_input_ids: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            chosen_input_ids (torch.LongTensor): Tensor of token ids for the chosen sequence.
            rejected_input_ids (torch.LongTensor): Tensor of token ids for the rejected sequence.
            chosen_attention_mask (Optional[torch.Tensor]): Tensor of attention masks for the chosen sequence.
            rejected_attention_mask (Optional[torch.Tensor]): Tensor of attention masks for the rejected sequence.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs instead of a tuple.

        Returns:
            torch.Tensor: Output tensor of value estimates.
        """
        # Get the model's outputs and extract the last hidden state
        outputs_chosen = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
        )
        last_hidden_states_chosen = outputs_chosen.last_hidden_state

        # Calculate the values and return the mean value for each sequence
        chosen_rewards = self.value_head(last_hidden_states_chosen).squeeze(-1)

        # Get the model's outputs and extract the last hidden state
        outputs_rejected = self.model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
        )
        last_hidden_states_rejected = outputs_rejected.last_hidden_state

        # Calculate the values and return the mean value for each sequence
        rejected_rewards = self.value_head(
            last_hidden_states_rejected).squeeze(-1)

        chosen_end_scores = []
        rejected_end_scores = []
        c_truncated_reward_list = []
        r_truncated_reward_list = []
        bs = chosen_input_ids.shape[0]
        for i in range(bs):

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item(
            ) if len(c_inds) > 0 else chosen_input_ids.shape[1]
            r_inds = (rejected_input_ids[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item(
            ) if len(r_inds) > 0 else rejected_input_ids.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_input_ids[i] !=
                              rejected_input_ids[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
            print(c_truncated_reward)
            print(r_truncated_reward)
            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])
            c_truncated_reward_list.append(c_truncated_reward)
            r_truncated_reward_list.append(r_truncated_reward)

        # Stack the end scores and return them
        c_truncated_reward_list = torch.stack(c_truncated_reward_list)
        r_truncated_reward_list = torch.stack(r_truncated_reward_list)

        chosen_end_scores = torch.stack(chosen_end_scores)
        rejected_end_scores = torch.stack(rejected_end_scores)
        # Calculate the loss
        loss = self.loss_fn(c_truncated_reward_list, r_truncated_reward_list)

        return RewardModelOutput(
            loss=loss,
            rewards_chosen=chosen_end_scores,
            rewards_rejected=rejected_end_scores,
        )


class RewardModel(nn.Module):
    """GPT Reward model.

    Args:
        model (str): Model name: 'opt', 'gpt2' or 'bloom'
        pretrained (str): Pretrained model name or path.
    """

    def __init__(self, pretrained: str = 'opt-125m'):
        super().__init__()

        # Instantiate tokenizer and model from pretrained checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            padding_side='left',
            truncation=True,
            padding=True,
        )
        self.model = AutoModel.from_pretrained(pretrained)

        # Set EOS token and padding token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.eos_token_id = 2
            # add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Instantiate model based on input string
        if 'opt' in pretrained:
            self.model = OPTModel.from_pretrained(pretrained)
        elif 'gpt2' in pretrained:
            self.model = GPT2Model.from_pretrained(pretrained)
        elif 'bloom' in pretrained:
            self.model = BloomModel.from_pretrained(pretrained)
        else:
            # If the model string is invalid, raise an error
            raise ValueError(
                "Invalid model name. Choose 'opt', 'gpt2' or 'bloom'.")

        # Get the model's config and create a value head
        self.config = self.model.config

        if 'opt' in pretrained:
            self.config.n_embd = self.config.word_embed_proj_dim
        self.value_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.PAD_ID = self.tokenizer.pad_token_id

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            input_ids (torch.LongTensor): Tensor of token ids for the input sequence.
            attention_mask (Optional[torch.Tensor]): Tensor indicating which tokens are padding tokens.
            return_dict (Optional[bool]): Whether to return a dictionary of outputs instead of a tensor.

        Returns:
            torch.Tensor: Output tensor of value estimates.
        """
        # If return_dict is not specified, use the default value
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Get the model's outputs and extract the last hidden state
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             return_dict=return_dict)
        last_hidden_states = outputs.last_hidden_state
        values = self.value_head(last_hidden_states)
        return values

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             use_cache=use_cache)
        hidden_states = outputs[0]
        values = self.value_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, 'prompt_length must be greater than 1 to help select the end score'
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forwad function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about
                # the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                'values': values,
                'chosen_end_scores': torch.stack(chosen_end_scores),
            }
