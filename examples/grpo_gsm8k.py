import re

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split('<answer>')[-1]
    answer = answer.split('</answer>')[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if '####' not in text:
        return None
    return text.split('####')[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split='train') -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            'prompt': [{
                'role': 'system',
                'content': SYSTEM_PROMPT
            }, {
                'role': 'user',
                'content': x['question']
            }],
            'answer':
            extract_hash_answer(x['answer'])
        })  # type: ignore
    return data  # type: ignore


def correctness_reward_func(prompts, completions, answer,
                            **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f'Question:\n{q}', f'\nAnswer:\n{answer[0]}',
          f'\nResponse:\n{responses[0]}',
          f'\nExtracted:\n{extracted_responses[0]}')
    return [
        2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>'
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count('<reasoning>\n') == 1:
        count += 0.125
    if text.count('\n</reasoning>\n') == 1:
        count += 0.125
    if text.count('\n<answer>\n') == 1:
        count += 0.125
        count -= len(text.split('\n</answer>\n')[-1]) * 0.001
    if text.count('\n</answer>') == 1:
        count += 0.125
        count -= (len(text.split('\n</answer>')[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]['content'] for completion in completions]
    return [count_xml(c) for c in contents]


dataset = get_gsm8k_questions()

model_name = 'meta-llama/Llama-3.2-1B-Instruct'

if 'Llama' in model_name:
    output_dir = 'outputs/Bellatrix-Grade-Math-GRPO'
    run_name = 'Bellatrix-Grade-Math-GRPO-gsm8k'
else:
    output_dir = 'outputs/Qwen-1.5B-GRPO'
    run_name = 'Qwen-1.5B-GRPO-gsm8k'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to='tensorboard',
    log_on_each_node=False,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj',
        'gate_proj'
    ],
    task_type='CAUSAL_LM',
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    device_map=None).to('cuda')

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(model=model,
                      processing_class=tokenizer,
                      reward_funcs=[
                          xmlcount_reward_func, soft_format_reward_func,
                          strict_format_reward_func, int_reward_func,
                          correctness_reward_func
                      ],
                      args=training_args,
                      train_dataset=dataset,
                      peft_config=peft_config)
trainer.train()
