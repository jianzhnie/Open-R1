
"""Push the details from a LightEval run to the Hub.

Usage:

python src/open_r1/utils/upload_details.py \
    --data_files {path_to_parquet_file} \
    --hub_repo_id {hub_repo_id} \
    --config_name {config_name}
"""

from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    data_files: List[str] = field(default_factory=list)
    hub_repo_id: str = None
    config_name: str = None


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if all(file.endswith('.json') for file in args.data_files):
        ds = load_dataset('json', data_files=args.data_files)
    elif all(file.endswith('.jsonl') for file in args.data_files):
        ds = load_dataset('json', data_files=args.data_files)
    else:
        ds = load_dataset('parquet', data_files=args.data_files)
    url = ds.push_to_hub(args.hub_repo_id,
                         config_name=args.config_name,
                         private=True)
    print(f'Dataset available at: {url}')


if __name__ == '__main__':
    main()
