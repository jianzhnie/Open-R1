import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from trl import (ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map,
                 get_peft_config, get_quantization_config)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

sys.path.append(os.getcwd())

from openr1.train.ppo.ppo_config import PPOConfig
from openr1.train.ppo.ppo_trainer import PPOTrainer
from openr1.utils.callbacks import get_callbacks
from openr1.utils.reward_funcs import (accuracy_reward, format_reward,
                                       get_cosine_scaled_reward,
                                       get_repetition_penalty_reward,
                                       reasoning_steps_reward)

logger = logging.getLogger(__name__)


@dataclass
class PPOScriptArguments(ScriptArguments):
    """Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda:
        ['accuracy', 'format', 'reasoning_steps', 'cosine'],
        metadata={
            'help':
            "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={'help': 'Minimum reward for wrong answers'},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={'help': 'Maximum reward for wrong answers'},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={'help': 'Minimum reward for correct answers'},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={'help': 'Maximum reward for correct answers'},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={'help': 'Maximum length for scaling'},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={'help': 'Number of n-grams for repetition penalty reward'},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={
            'help':
            'Maximum (negative) penalty for for repetition penalty reward'
        },
    )


SYSTEM_PROMPT = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant '
    'first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning '
    'process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., '
    '<think> reasoning process here </think><answer> answer here </answer>')


def main(script_args: PPOScriptArguments, training_args: PPOConfig,
         model_args: ModelConfig):

    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        +
        f' distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Data parameters {training_args}')

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f'Checkpoint detected, resuming training at {last_checkpoint=}.')

    ################
    # Model & Tokenizer
    ################
    logger.info('*** Initializing model kwargs ***')
    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in [
        'auto', None
    ] else getattr(torch, model_args.torch_dtype))
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map()
        if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side='left',
        trust_remote_code=model_args.trust_remote_code)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1)
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path,
    #     trust_remote_code=model_args.trust_remote_code,
    #     num_labels=1)
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code)

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code)
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name,
                           name=script_args.dataset_config)
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        'accuracy':
        accuracy_reward,
        'format':
        format_reward,
        'reasoning_steps':
        reasoning_steps_reward,
        'cosine':
        get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        'repetition_penalty':
        get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }
    reward_funcs = [
        REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs
    ]

    # Format into conversation
    def make_conversation(example, tokenizer):

        prompt = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': example['problem']
            },
        ]
        return {'prompt': prompt}

    # 处理数据集 - 修复这里的调用
    dataset = dataset.map(
        function=make_conversation,
        fn_kwargs={'tokenizer': tokenizer},  # 传入tokenizer参数
        desc='Processing dataset',
    )
    for split in dataset:
        if 'messages' in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns(
                ['messages', 'problem'])

    # 打印处理后的样本示例
    print('\nProcessed example:')
    print(dataset[script_args.dataset_train_split][0])

    ################
    #  Initialize the PPO trainer
    ################
    trainer = PPOTrainer(
        args=training_args,
        model=policy,
        ref_model=ref_policy,
        reward_funcs=reward_funcs,
        value_model=value_model,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        # callbacks=get_callbacks(training_args, model_args),
        peft_config=peft_config,
    )

    ###############
    # Training loop
    ###############

    logger.info('*** Train ***')
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics['train_samples'] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Save and push to hub
    ##################################
    # Save model and create model card
    ##################################
    logger.info('*** Save model ***')
    trainer.save_model(training_args.output_dir)
    logger.info(f'Model saved to {training_args.output_dir}')
    trainer.save_model(training_args.output_dir)

    # Save everything else on main process
    kwargs = {
        'dataset_name': script_args.dataset_name,
        'tags': ['open-r1'],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        metrics['eval_samples'] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info('Pushing to hub...')
        trainer.push_to_hub(**kwargs)


if __name__ == '__main__':
    parser = TrlParser((PPOScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
