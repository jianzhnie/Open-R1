from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            'help':
            ('Name(s) of modules apart from LoRA layers to be set as trainable '
             'and saved in the final checkpoint. '
             'Use commas to separate multiple modules.')
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={'help': 'The intrinsic dimension for LoRA fine-tuning.'},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={
            'help':
            'The scale factor for LoRA fine-tuning (default: lora_rank * 2).'
        },
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'Dropout rate for the LoRA fine-tuning.'},
    )
    lora_target: str = field(
        default='all',
        metadata={
            'help':
            ('Name(s) of target modules to apply LoRA. '
             'Use commas to separate multiple modules. '
             'Use `all` to specify all the linear modules. '
             'LLaMA choices: [`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], '
             'BLOOM & Falcon & ChatGLM choices: [`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`], '
             'Baichuan choices: [`W_pack`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], '
             'Qwen choices: [`c_attn`, `attn.c_proj`, `w1`, `w2`, `mlp.c_proj`], '
             'InternLM2 choices: [`wqkv`, `wo`, `w1`, `w2`, `w3`], '
             'Others choices: the same as LLaMA.')
        },
    )
    lora_bias: Literal['none', 'all', 'lora_only'] = field(
        default='none',
        metadata={'help': 'Whether or not to use bias for LoRA layers.'},
    )
    use_qlora: bool = field(
        default=False,
        metadata={
            'help':
            'Whether or not to use the LoRA+ method for learning rate scaling.'
        },
    )


@dataclass
class QuantArguments:
    # 使用8-bit的adam，是否可以调整为LION或Sophia，甚至deepspeed还提供了多个1-bit优化器选择
    adam8bit: bool = field(default=False, metadata={'help': 'Use 8-bit adam.'})
    # 使用的位宽，默认为4。
    quant_bit: Optional[int] = field(
        default=4,
        metadata={
            'help':
            'The number of bits to quantize the model using bitsandbytes.'
        },
    )
    llm_int8_threshold: Optional[float] = field(
        default=6.0,
        metadata={
            'help':
            'The threshold for int8 quantization. Only applicable for LLMs with int8 weights.'
        },
    )
    llm_int8_has_fp16_weight: Optional[bool] = field(
        default=False,
        metadata={
            'help':
            'Whether to use fp16 weights for int8 training. Only applicable for LLMs with fp16 weights.'
        },
    )
    # 量化类型，可以选择`fp4`或`nf4`
    quant_type: Literal['fp4', 'nf4'] = field(
        default='nf4',
        metadata={
            'help': 'Quant data type to use. Should be one of `fp4` or `nf4`.'
        },
    )
    # 是否使用二次量化
    double_quant: bool = field(
        default=True,
        metadata={
            'help': 'Compress the quant statistics through double quant.'
        },
    )
    quant_device_map: Optional[Literal['auto']] = field(
        default=None,
        metadata={
            'help':
            'Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0.'
        },
    )
    export_quant_bit: Optional[int] = field(
        default=None,
        metadata={
            'help': 'The number of bits to quantize the exported model.'
        },
    )
    export_quant_dataset: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Path to the dataset or dataset name to use in quantizing the exported model.'
        },
    )
    export_quant_nsamples: int = field(
        default=128,
        metadata={'help': 'The number of samples used for quant.'},
    )
    export_quant_maxlen: int = field(
        default=1024,
        metadata={
            'help': 'The maximum length of the model inputs used for quant.'
        },
    )

    def __post_init__(self):
        if self.quant_bit is not None:
            assert self.quant_bit in [
                4,
                8,
            ], 'We only accept 4-bit or 8-bit quant.'
        if self.quant_type is not None:
            assert self.quant_type in [
                'nf4',
                'fp4',
            ], 'We only accept `nf4` or `fp4` quant type.'
        assert self.export_quant_bit in [
            None,
            8,
            4,
            3,
            2,
        ], 'We only accept 2/3/4/8-bit quantization.'

        if self.export_quant_bit is not None and self.export_quant_dataset is None:
            raise ValueError(
                'Quantization dataset is necessary for exporting.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinetuningArguments(
        LoraArguments,
        QuantArguments,
):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
