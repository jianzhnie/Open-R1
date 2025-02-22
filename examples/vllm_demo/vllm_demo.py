# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is',
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
model_nameor_path = 'facebook/opt-125m'
model_nameor_path = '/home/robin/hf_hub/models/Qwen/Qwen2.5-1.5B/'
llm = LLM(model=model_nameor_path,
          tensor_parallel_size=1,
          trust_remote_code=True,
          max_model_len=10000,
          dtype='half',
          gpu_memory_utilization=0.5,
          block_size=32)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
