import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather, gather_object, is_peft_model
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
    BaseImageProcessor, DataCollatorWithPadding, FeatureExtractionMixin,
    GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin,
    Trainer, TrainerCallback, TrainerControl, is_wandb_available)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import (CallbackHandler, ExportableState,
                                           PrinterCallback)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import is_peft_available
from trl.core import masked_mean, masked_whiten
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState, batch_generation, disable_dropout_in_model, exact_div,
    first_true_indices, forward, generate_model_card, get_comet_experiment_url,
    get_reward, log_table_to_comet_experiment, peft_module_casting_to_bf16,
    prepare_deepspeed, print_rich_table, truncate_response)

from openr1.train.ppo.configs import PPOConfig
from openr1.utils.utils import selective_log_softmax

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):

    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model,
                                       value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class PPOTrainer(Trainer):
    _tag_names = ['trl', 'ppo']

    def __init__(
        self,
        args: PPOConfig,
        policy_model: Union[str, PreTrainedModel],
        ref_model: Optional[Union[str, PreTrainedModel]],
        value_model: Optional[Union[str, PreTrainedModel]],
        reward_funcs: RewardFunc,
        reward_func_names: List[str],
        processing_class: Optional[Union[PreTrainedTokenizerBase,
                                         BaseImageProcessor,
                                         FeatureExtractionMixin,
                                         ProcessorMixin]],
        reward_processing_classes: Optional[Union[
            PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer,
                          torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional['PeftConfig'] = None,
    ) -> None:

        self.args = args
        self.processing_class = processing_class

        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(policy_model, str):
            model_name_or_path = policy_model
            torch_dtype = model_init_kwargs.get('torch_dtype')
            if (isinstance(torch_dtype, torch.dtype) or torch_dtype == 'auto'
                    or torch_dtype is None):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs['torch_dtype'] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs['use_cache'] = (
                False if args.gradient_checkpointing else
                model_init_kwargs.get('use_cache'))
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                policy_model, **model_init_kwargs)
        else:
            model_name_or_path = policy_model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    'You passed `model_init_kwargs` to the `PPOConfig`, but your model is already instantiated. '
                    'This argument can only be used when the `model` argument is a string.'
                )
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.policy_model = self._enable_gradient_checkpointing(
                self.policy_model, args)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError(
                'You cannot set both `stop_token` and `stop_token_id`.')
        elif args.stop_token:
            if args.stop_token == 'eos':
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, 'is_loaded_in_4bit',
                                     False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(
            self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        # Reference model
        self.kl_coef = args.kl_coef
        if self.kl_coef == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **model_init_kwargs)
        elif is_peft_model:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(self.policy_model)

        # Value Model
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, **model_init_kwargs, num_labels=1)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_funcs[i] = reward_func
        self.reward_funcs = reward_funcs
        self.reward_func_names = reward_func_names

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                    f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights,
                                               dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs),
                                             dtype=torch.float32)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                policy_model.config._name_or_path,
                padding_side='left',
                trust_remote_code=model_init_kwargs.trust_remote_code)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    'The number of reward processing classes must match the number of reward functions.'
                )

        for i, (reward_processing_class, reward_func) in enumerate(
                zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token)
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs *
                                      self.train_dataset_len)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (args.per_device_train_batch_size *
                                 args.gradient_accumulation_steps *
                                 args.num_mini_batches)
        args.micro_batch_size = int(args.per_device_train_batch_size *
                                    args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches,
            '`batch_size` must be a multiple of `num_mini_batches`')
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches,
            '`local_batch_size` must be a multiple of `num_mini_batches`')
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f'Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening'
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes /
            args.batch_size)  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(
            time_tensor,
            0).item()  # avoid different timestamps across processes
        args.run_name = f'{args.exp_name}__{args.seed}__{time_int}'
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [
                self.policy_model,
                self.ref_model,
                self.value_model,
        ]:

            if module is not None:
                disable_dropout_in_model(module)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                disable_dropout_in_model(module)

        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        # trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(self.callbacks, self.model,
                                                self.processing_class,
                                                self.optimizer,
                                                self.lr_scheduler)
        self.add_callback(PrinterCallback if self.args.
                          disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state,
                                            'deepspeed_plugin',
                                            None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, 'fsdp_plugin',
                                       None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, 'add_model_tags'):
            self.model.add_model_tags(self._tag_names)

        #########
        # setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=
            True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True)

        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(reward_func, nn.Module):
                    self.reward_funcs[i] = prepare_deepspeed(
                        reward_func, args.per_device_train_batch_size,
                        args.fp16, args.bf16)

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        'No reference model and model is not a Peft model.')
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size,
                    args.fp16, args.bf16)
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        'No reference model and model is not a Peft model.')
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)
                for i, reward_func in enumerate(self.reward_funcs):
                    if isinstance(reward_func, nn.Module):
                        self.reward_funcs[i] = reward_func.to(
                            self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def _enable_gradient_checkpointing(self, model: PreTrainedModel,
                                       args: PPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = ('use_reentrant' not in gradient_checkpointing_kwargs
                         or gradient_checkpointing_kwargs['use_reentrant'])

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft
        adapter manipulation)."""
        with (self.accelerator.unwrap_model(
                self.model.policy).disable_adapter() if self.is_peft_model
              and not self.ref_adapter_name else nullcontext()):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name
                                              or 'default')

    def save_model(self,
                   output_dir: Optional[str] = None,
                   _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print('===training policy===')
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches,
                       args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps *
                                                     args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps *
                                                  args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps *
                                                  args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            prompts = [x['prompt'] for x in data]
            with torch.no_grad():
                prompts_text = [
                    maybe_apply_chat_template(example,
                                              self.processing_class)['prompt']
                    for example in data
                ]
                keys = [
                    key for key in data[0]
                    if key not in ['prompt', 'completion']
                ]
                extra_data = {
                    key: [example[key] for example in data]
                    for key in keys
                }
                query_inputs = self.processing_class(prompts_text,
                                                     return_tensors='pt',
                                                     padding=True,
                                                     padding_side='left',
                                                     add_special_tokens=False)
                query_inputs = super()._prepare_inputs(query_inputs)
                queries, query_mask = query_inputs['input_ids'], query_inputs[
                    'attention_mask']

                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                reward_func_metrics = {}
                for i, func_name in enumerate(self.reward_func_names):
                    reward_func_metrics[func_name] = []

                with unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=self.args.
                        ds3_gather_for_generation) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0],
                               args.local_rollout_forward_batch_size):

                    prompts_batch = prompts[i:i + args.
                                            local_rollout_forward_batch_size]
                    reward_extra_data = {
                        key:
                        extra_data[key][i:i +
                                        args.local_rollout_forward_batch_size]
                        for key in extra_data
                    }
                    query = queries[i:i +
                                    args.local_rollout_forward_batch_size]
                    query_response = query_responses[
                        i:i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i:i +
                                     args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response,
                                                 processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response,
                                             processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1:-1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id,
                            response)

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1)
                    sequence_length = first_true_indices(
                        postprocessed_response ==
                        processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(
                        model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response,
                        processing_class.pad_token_id, context_length)
                    value = full_value[:, context_length - 1:-1].squeeze(-1)

                    # Decode the generated completions
                    completions_text = self.processing_class.batch_decode(
                        postprocessed_response, skip_special_tokens=True)
                    query_text = self.processing_class.batch_decode(
                        query, skip_special_tokens=True)

                    if_conversation = is_conversational(data[0])
                    if if_conversation:
                        completions = []
                        for prompt, completion in zip(prompts_batch,
                                                      completions_text):
                            bootstrap = prompt.pop()['content'] if prompt[-1][
                                'role'] == 'assistant' else ''
                            completions.append([{
                                'role':
                                'assistant',
                                'content':
                                bootstrap + completion
                            }])
                    else:
                        completions = completions_text

                    # Get the reward
                    rewards_per_func = torch.zeros(len(completions),
                                                   len(self.reward_funcs),
                                                   device=device)

                    for i, (reward_func, reward_processing_class) in enumerate(
                            zip(self.reward_funcs,
                                self.reward_processing_classes)):

                        # Module instead of PretrainedModel for compat with compiled models
                        if isinstance(reward_func, nn.Module):

                            with torch.inference_mode():
                                _, score, _ = get_reward(
                                    reward_func, postprocessed_query_response,
                                    reward_processing_class.pad_token_id,
                                    context_length)
                        else:

                            output_reward_func = reward_func(
                                prompts=query_text,
                                completions=completions,
                                **reward_extra_data)
                            score = torch.tensor(output_reward_func,
                                                 dtype=torch.float32,
                                                 device=device)

                        rewards_per_func[:, i] = score

                    # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
                    # completions may be distributed across processes
                    rewards_per_func = gather(rewards_per_func)

                    for i, func_name in enumerate(self.reward_func_names):
                        reward_func_metrics[func_name].append(
                            rewards_per_func[:, i])

                    # Apply weights to each reward function's output and sum
                    score = (rewards_per_func *
                             self.reward_weights.to(device).unsqueeze(0)).sum(
                                 dim=1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)

                reward_func_metrics = {
                    func_name: torch.cat(reward_func_metrics[func_name], 0)
                    for func_name in self.reward_func_names
                }

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score,
                     unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(
                    postprocessed_responses ==
                    self.processing_class.eos_token_id,
                    dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                accelerator.print(
                    f'{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}'
                )

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1],
                                             device=responses.device).repeat(
                                                 responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask,
                                             INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask,
                                                 INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (
                    sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0),
                                            device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1),
                                         sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards,
                                            mask=~padding_mask_p1,
                                            shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:,
                                        t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:,
                                                                             t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size,
                                              args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                            0, args.local_mini_batch_size,
                            args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[
                                micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[
                                micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(
                                model, mb_query_responses,
                                processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1:-1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(
                                logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds],
                                INVALID_LOGPROB)
                            vpred = vpred_temp[:, context_length -
                                               1:-1].squeeze(-1)
                            vpred = torch.masked_fill(
                                vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(
                                vf_loss_max,
                                ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(),
                                ~padding_mask_p1[micro_batch_inds])
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(
                                ratio, 1.0 - args.cliprange,
                                1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(
                                pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(),
                                    ~padding_mask[micro_batch_inds])
                                prob_dist = torch.nn.functional.softmax(logits,
                                                                        dim=-1)
                                entropy = torch.logsumexp(
                                    logits, dim=-1) - torch.sum(
                                        prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = (pg_clipfrac)
                                pg_loss_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = (vf_clipfrac)
                                entropy_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = entropy.mean(
                                    )
                                ratio_stats[
                                    ppo_epoch_idx, minibatch_idx,
                                    gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output,
                        vpred_temp,
                        logits,
                        new_logprobs,
                        vpred,
                        vpredclipped,
                        vf_losses1,
                        vf_losses2,
                        vf_loss,
                        vf_clipfrac,
                        logprobs_diff,
                        ratio,
                        pg_losses,
                        pg_losses2,
                        pg_loss_max,
                        pg_loss,
                        loss,
                        pg_clipfrac,
                        prob_dist,
                        entropy,
                        approxkl,
                        mb_return,
                        mb_advantage,
                        mb_values,
                        mb_responses,
                        mb_query_responses,
                        mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                mean_sequence_length = sequence_lengths.float().mean()
                mean_reward_per_func = {
                    func_name: reward_func_metrics[func_name].mean()
                    for func_name in self.reward_func_names
                }
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics['eps'] = eps
                for key, val in mean_reward_per_func.items():
                    metrics['reward_funcs/' + str(key)] = val.item()
                metrics[
                    'reward_funcs/sequence_length'] = mean_sequence_length.item(
                    )
                metrics['objective/kl'] = self.accelerator.gather_for_metrics(
                    mean_kl).mean().item()
                metrics[
                    'objective/entropy'] = self.accelerator.gather_for_metrics(
                        mean_entropy).mean().item()
                metrics['objective/non_score_reward'] = (
                    self.accelerator.gather_for_metrics(
                        mean_non_score_reward).mean().item())
                metrics[
                    'objective/rlhf_reward'] = self.accelerator.gather_for_metrics(
                        rlhf_reward).mean().item()
                metrics[
                    'objective/scores'] = self.accelerator.gather_for_metrics(
                        scores.mean()).mean().item()
                metrics[
                    'policy/approxkl_avg'] = self.accelerator.gather_for_metrics(
                        approxkl_stats).mean().item()
                metrics[
                    'policy/clipfrac_avg'] = self.accelerator.gather_for_metrics(
                        pg_clipfrac_stats).mean().item()
                metrics[
                    'loss/policy_avg'] = self.accelerator.gather_for_metrics(
                        pg_loss_stats).mean().item()
                metrics[
                    'loss/value_avg'] = self.accelerator.gather_for_metrics(
                        vf_loss_stats).mean().item()
                metrics[
                    'val/clipfrac_avg'] = self.accelerator.gather_for_metrics(
                        vf_clipfrac_stats).mean().item()
                metrics[
                    'policy/entropy_avg'] = self.accelerator.gather_for_metrics(
                        entropy_stats).mean().item()
                metrics['val/ratio'] = self.accelerator.gather_for_metrics(
                    ratio_stats).mean().item()
                metrics['val/ratio_var'] = self.accelerator.gather_for_metrics(
                    ratio_stats).var().item()
                metrics['val/num_eos_tokens'] = (
                    responses == processing_class.eos_token_id).sum().item()
                metrics['lr'] = self.lr_scheduler.get_last_lr()[0]
                metrics['episode'] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            if (update + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(
                        self.args.output_dir,
                        f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'))
            self.control = self.callback_handler.on_step_end(
                args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (
                    update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        device = self.accelerator.device
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                prompts = [x['prompt'] for x in batch]
                prompts_text = [
                    maybe_apply_chat_template(example,
                                              self.processing_class)['prompt']
                    for example in batch
                ]
                keys = [
                    key for key in batch[0]
                    if key not in ['prompt', 'completion']
                ]
                extra_data = {
                    key: [example[key] for example in batch]
                    for key in keys
                }
                query_inputs = self.processing_class(prompts_text,
                                                     return_tensors='pt',
                                                     padding=True,
                                                     padding_side='left',
                                                     add_special_tokens=False)
                query_inputs = super()._prepare_inputs(query_inputs)
                query, query_mask = query_inputs['input_ids'], query_inputs[
                    'attention_mask']

                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id,
                            response)
                    table['query'].extend(
                        gather_object(
                            processing_class.batch_decode(
                                query, skip_special_tokens=True)))
                    table['model response'].extend(
                        gather_object(
                            processing_class.batch_decode(
                                postprocessed_response)))

                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1)

                    # Decode the generated completions
                    completions_text = self.processing_class.batch_decode(
                        postprocessed_response, skip_special_tokens=True)
                    query_text = self.processing_class.batch_decode(
                        query, skip_special_tokens=True)

                    if_conversation = is_conversational(batch[0])
                    if if_conversation:
                        completions = []
                        for prompt, completion in zip(prompts,
                                                      completions_text):
                            bootstrap = prompt.pop()['content'] if prompt[-1][
                                'role'] == 'assistant' else ''
                            completions.append([{
                                'role':
                                'assistant',
                                'content':
                                bootstrap + completion
                            }])
                    else:
                        completions = completions_text

                    rewards_per_func = torch.zeros(len(completions),
                                                   len(self.reward_funcs),
                                                   device=device)

                    for i, (reward_func, reward_processing_class) in enumerate(
                            zip(self.reward_funcs,
                                self.reward_processing_classes)):
                        if isinstance(reward_func, nn.Module):
                            # Module instead of PretrainedModel for compat with compiled models

                            with torch.inference_mode():
                                _, score, _ = get_reward(
                                    reward_func, postprocessed_query_response,
                                    reward_processing_class.pad_token_id,
                                    context_length)

                        else:
                            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                            output_reward_func = reward_func(
                                prompts=query_text,
                                completions=completions,
                                **extra_data)
                            score = torch.tensor(output_reward_func,
                                                 dtype=torch.float32,
                                                 device=device)

                        rewards_per_func[:, i] = score

                    # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
                    # completions may be distributed across processes
                    rewards_per_func = gather(rewards_per_func)

                    # Apply weights to each reward function's output and sum
                    score = (rewards_per_func *
                             self.reward_weights.to(device).unsqueeze(0)).sum(
                                 dim=1)

                    table['score'].extend(
                        self.accelerator.gather_for_metrics(
                            score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0:0 + 5])
            if 'wandb' in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({'completions': wandb.Table(dataframe=df)})

            if 'comet_ml' in args.report_to:
                log_table_to_comet_experiment(
                    name='completions.csv',
                    table=df,
                )

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """Creates a draft of a model card using the information available to
        the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, '_name_or_path') and not os.path.isdir(
                self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, 'unsloth_version'):
            tags.append('unsloth')

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url()
            if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name='PPO',
            trainer_citation=citation,
            paper_title='Fine-Tuning Language Models from Human Preferences',
            paper_id='1909.08593',
        )

        model_card.save(os.path.join(self.args.output_dir, 'README.md'))
