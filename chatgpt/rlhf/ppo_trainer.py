import os
from collections import deque
from typing import Deque, Optional, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtyping import TensorType

from chatgpt.buffer.replay_buffer import ExperienceDataset, Memory
from chatgpt.buffer.rollout import BaseRolloutStore
from chatgpt.dataset.prompt_dataset import PromptDataset
from chatgpt.rlhf.actor_critic import ActorCritic
from chatgpt.rlhf.reward_model import RewardModel
from chatgpt.utils.modeling import flatten_dict, get_tensor_stats, whiten

"""
train()
┌─────────────────────────────┐
│                             │◄─────────────────────────┐
│                             │                          │
│      ┌─────────────┐        │                          │
│      │ user input  │        │                          │ learn()
│      └─────┬───────┘        │             ┌────────────┴─────────────┐
│            │                │             │                          │
│            │                │             │       ┌────────┐         │
│            │                │             │   ┌───│ Update │──┐      │
│            │                │             │   │   └────▲───┘  │      │
│   ┌────────▼────────────┐   │             │   │        │      │      │
│   │  Actor (LLM Model)  │   │             │   │     ┌──┴───┐  │      │
│   └────────┬────────────┘   │             │   │     │ PPO  │  │      │
│            │                │             │   │     └▲────▲┘  │      │
│            │                │             │   │      │    │   │      │
│            │                │             │   │      │    │   │      │
│    ┌───────▼──────┐         │             │ ┌─▼──────┴┐ ┌─┴───▼──┐   │
│    │ Reward Model │         │             │ │  Actor  │ │ Critic │   │
│    └──────────────┘         │             │ └─────────┘ └────────┘   │
│                             │             │                          │
│                             │ x Episodes  └─────────────▲────────────┘
└───────────────┬─────────────┘                           │   x Epochs
                │ store N Examples per Timestep           │
         ┌──────▼──────┐                                  │
         │             │                                  │
         │  Memories   ├──────────────────────────────────┘
         │             │ (update timesteps x N Examples)
         └─────────────┘
"""  # noqa W291


class PPOTrainer:

    def __init__(
        self,
        prompt_data_path: str,
        pretrained_model: str = 'facebook/opt-125m',
        num_episodes: int = 1,
        ppo_epochs: int = 1,
        batch_size: int = 32,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        beta_s: float = 0.01,
        actor_eps_clip: float = 0.2,
        critic_eps_clip: float = 0.2,
        checkpoint_episode: int = 1,
        work_dirs: str = 'work_dirs',
        device: str = 'cpu',
        debug: bool = False,
    ) -> None:

        # Those value can be changed
        self.num_episodes = num_episodes
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.beta_s = beta_s
        self.actor_eps_clip = actor_eps_clip
        self.critic_eps_clip = critic_eps_clip
        self.eps = 1e-8
        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.checkpoint_episode = checkpoint_episode
        self.device = device
        self.debug = debug
        self.model_folder = os.path.join(work_dirs, 'checkpoints')

        # initialize agent-critic
        self.actor_critic = ActorCritic(pretrained=pretrained_model,
                                        debug=False).to(device)
        # initialize optimizers
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(),
                                    lr=actor_lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(),
                                     lr=critic_lr)
        # initialize reward model
        self.reward_model = RewardModel(pretrained=pretrained_model).to(device)
        # initialize examples sampler
        self.prompt_dataset = PromptDataset(data_path=prompt_data_path,
                                            split='train')
        self.prompt_dataloader = DataLoader(self.prompt_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)

    def save_checkpoint(
        self,
        current_episode: int,
        max_episode: int,
        path: str,
    ) -> None:

        print(f'Saving checkpoint for episode {current_episode+1}..')
        # if the checkpoint already exists remove it.
        # Deepspeed checkpoints are already directories and will be overwritten
        critic_model_path = os.path.join(path, 'critic')
        critic_file_name = os.path.join(critic_model_path,
                                        current_episode + '.tar')
        if not os.path.exists(critic_model_path):
            os.makedirs(critic_model_path)

        # save the checkpoint
        critic_checkpoint_dict = {
            'episode': current_episode,
            'critic_state_dict': self.actor_critic.critic.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
        }

        torch.save(critic_checkpoint_dict, critic_file_name)

        # if the checkpoint already exists remove it.
        # Deepspeed checkpoints are already directories and will be overwritten
        actor_model_path = os.path.join(path, 'actor')
        actor_file_name = os.path.join(critic_model_path,
                                       current_episode + '.tar')
        if not os.path.exists(actor_model_path):
            os.makedirs(actor_model_path)

        # save the checkpoint
        actor_checkpoint_dict = {
            'episode': current_episode,
            'actor_state_dict': self.actor_critic.actor.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
        }
        torch.save(actor_checkpoint_dict, actor_file_name)

    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:

        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print('Start to Learn...')
        # create dataset from memories
        dataset = ExperienceDataset(memories)

        dataloader = DataLoader(dataset, batch_size=8)
        # train agent-critic
        self.actor_critic.train()
        for epoch in range(self.ppo_epochs):
            for i, batch in enumerate(dataloader):
                (
                    states_actor,
                    old_actions,
                    old_values,
                    rewards,
                    old_actions_log_probs,
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                ) = [tensor.to(self.device) for tensor in batch]

                if self.debug:
                    print(f'#########################################'
                          f' batch from memories {i} \n '
                          f'#########################################'
                          f'states_actor {states_actor.shape} \n'
                          f'old_actions {old_actions.shape} \n'
                          f'old_values {old_values.shape} \n'
                          f'rewards {rewards.shape} \n'
                          f'old_actions_log_probs '
                          f'{old_actions_log_probs.shape}\n'
                          f'sequences_actor {sequences_actor.shape} \n'
                          f'sequences_mask_actor '
                          f'{sequences_mask_actor.shape} \n'
                          f'sequences_critic {sequences_critic.shape} \n'
                          f'sequences_mask_critic '
                          f'{sequences_mask_critic.shape} \n'
                          f'action_len_actor {action_len_actor} \n'
                          f'action_len_critic {action_len_critic} \n'
                          f'#########################################')

                # get actor critic new probabilities and values
                actions_logits, values = self.actor_critic.forward(
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                )

                # get action log prob
                actions_prob = (torch.softmax(actions_logits,
                                              dim=-1).max(dim=-1).values)
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob *
                     (old_actions_log_probs - actions_log_prob)).sum(
                         dim=-1).mean())
                # compute PPO Loss -- Whan dimensions are different
                # (especially the values and the probs are
                #  multiplied directly with the reward)
                ratios = (actions_log_prob - old_actions_log_probs).exp()
                advantages = rewards - old_values[:, -1]

                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps)
                surr1 = advantages * ratios
                surr2 = (torch.clamp(ratios, 1 - self.actor_eps_clip,
                                     1 + self.actor_eps_clip) * advantages)
                policy_loss = -torch.min(surr1,
                                         surr2) - self.beta_s * entropies
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError('Loss is nan')
                print('loss', loss.item())

                if self.debug:
                    print('values', values)
                    print('old_values', old_values)
                    print('rewards', rewards)
                    print('ratios', ratios)
                    print('advantages', advantages)
                    print('entropies', entropies)

                # update actor with loss
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                # compute value loss
                # the loss is the distance between the rewards and the values
                # I want this distance to be small so that values are
                # representative of the rewards, for this reason i took the
                # maximum between the two.
                # The clip is limiting the slew-rate of values_loss_clipped
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -self.critic_eps_clip, self.critic_eps_clip)
                value_loss1 = (value_loss_clipped - rewards)**2
                value_loss2 = (values - rewards)**2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError('Value loss is nan')
                print('value_loss', value_loss.item())

                # upate critic with loss
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

                # print iteration info
                print(
                    f'Epoch {epoch+1}/{self.epochs}',
                    f'Step {i+1}/{int(len(dataloader) / self.batch_size)}',
                    f'Loss {loss.detach().cpu().item():.4f}',
                    f'Value Loss {value_loss.detach().cpu().item():.4f}',
                )
        self.actor_critic.eval()
        print('End Learning')
        return policy_loss.item(), value_loss.item(), kl_div_loss.item()

    def train(self) -> None:
        print('Start RL Training')
        # initialize memories
        memories = deque([])
        # initialize counters
        cnt_timesteps = 0
        cnt_learn_iter = 0

        max_timesteps = len(self.prompt_dataloader) * self.num_episodes
        update_timesteps = max_timesteps * self.ppo_epochs
        # loop over episodes and timesteps
        self.actor_critic.eval()
        for episode in range(self.num_episodes):
            for step, inputs in enumerate(self.prompt_dataloader):
                print(
                    f'Episode: {episode + 1}/{self.num_episodes}',
                    f'Step: {cnt_timesteps + 1}/{max_timesteps}',
                    f'Learning Cnt: {cnt_learn_iter + 1}/{update_timesteps}',
                )
                # counter used to count timesteps into memory
                cnt_timesteps += 1
                # tokenize examples for the actor
                tok_inputs_act = self.actor_critic.actor.tokenizer(
                    inputs, padding=True, return_tensors='pt', truncation=True)

                # states are [batch_size, seq_len_of_states]
                states_actor = tok_inputs_act['input_ids'].to(self.device)
                states_mask_actor = tok_inputs_act['attention_mask'].to(
                    self.device)

                # tokenize examples for the critic
                tok_inputs_crt = self.actor_critic.critic.tokenizer(
                    inputs, padding=True, return_tensors='pt', truncation=True)

                # states are [batch_size, seq_len_of_states]
                states_critic = tok_inputs_crt['input_ids'].to(self.device)

                # generate sequences of actions and values
                (
                    actions,
                    actions_logits,
                    values,
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                ) = self.actor_critic.generate(states_actor, states_mask_actor,
                                               states_critic)

                # from action logits to action log probs
                action_prob = (torch.softmax(actions_logits,
                                             dim=-1).max(dim=-1).values)
                actions_log_probs = torch.log(action_prob + self.eps)

                reward_sequence = sequences_critic
                reward_mask = sequences_mask_actor

                # compute rewards
                rewards = self.reward_model.forward(
                    reward_sequence,
                    reward_mask,
                )
                rewards = rewards[:, -action_len_critic:]

                # store memories of the episode / timestep
                for i in range(states_actor.shape[0]):
                    memories.append(
                        Memory(
                            states_actor[i, :].detach().cpu(),
                            actions[i, :].detach().cpu(),
                            values[i, :].detach().cpu(),
                            rewards[i, :].detach().cpu(),
                            actions_log_probs[i, :].detach().cpu(),
                            sequences_actor[i, :].detach().cpu(),
                            sequences_mask_actor[i, :].detach().cpu(),
                            sequences_critic[i, :].detach().cpu(),
                            sequences_mask_critic[i, :].detach().cpu(),
                            int(action_len_actor),
                            int(action_len_critic),
                        ))
                # learn from memories
                if (cnt_timesteps % update_timesteps
                        == 0) and (cnt_timesteps != 0):
                    print('len memories', len(memories))
                    # self.conversation_log.show(cnt_learn_iter)
                    self.learn(memories)
                    mean_reward = sum([m.rewards[-1]
                                       for m in memories]) / len(memories)
                    print(f'Mean Reward: {mean_reward}')
                    memories.clear()
                    cnt_timesteps = 0
                    cnt_learn_iter += 1

            # save checkpoints
            if (episode % self.checkpoint_episode == 0) and (episode != 0):
                self.save_checkpoint(current_episode=episode,
                                     max_episode=self.num_episodes,
                                     path=self.model_folder)

        print('End RL Training')

    def compute_rewards(
        self,
        prompts,
        log_probs: TensorType['batch_size', 'response_size'],
        ref_log_probs: TensorType['batch_size', 'response_size'],
        reward_score: TensorType['batch_size', 'response_size'],
        action_mask: TensorType['batch_size', 'response_size'],
    ):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def get_advantages_and_returns(
        self,
        values: TensorType['batch_size', 'response_size'],
        rewards: TensorType['batch_size', 'response_size'],
        response_length: int,
        use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and
        values. Calculated as in the original PPO paper:
        https://arxiv.org/abs/1707.06347 Note that rewards may include a KL
        divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def actor_loss_fn(
        self,
        logprobs: TensorType['batch_size', 'response_size'],
        old_logprobs: TensorType['batch_size', 'response_size'],
        advantages: TensorType['batch_size', 'response_size'],
        mask: TensorType['batch_size', 'response_size'],
    ):
        # policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(
        self,
        values: TensorType['batch_size', 'response_size'],
        old_values: TensorType['batch_size', 'response_size'],
        returns: TensorType['batch_size', 'response_size'],
        mask: TensorType['batch_size', 'response_size'],
    ):
        # value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_loss(
        self,
        logprobs: TensorType['batch_size', 'response_size'],
        values: TensorType['batch_size', 'response_size'],
        old_logprobs: TensorType['batch_size', 'response_size'],
        old_values: TensorType['batch_size', 'response_size'],
        advantages: TensorType['batch_size', 'response_size'],
        returns: TensorType['batch_size', 'response_size'],
        mask: TensorType['batch_size', 'response_size'],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask)**2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(),
                        clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=n / mask.numel(),
        )

        return loss, flatten_dict(stats)

    def add_prompt_rollout(self, pipeline: BaseRolloutStore):
        """Add a prompt pipeline dataloader to a trainer instance for the
        `make_experience` stage."""
        prompt_dataloader = pipeline.create_loader(self.chunk_size,
                                                   shuffle=True)
        self.prompt_iterator = iter(prompt_dataloader)
