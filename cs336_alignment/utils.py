import json 

import torch


dtype_mapping= {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# return a list of dics
def load_dataset(data_file: str, data_type: str='train'):
    with open(data_file, 'r') as f:
        if data_type in ['train', 'val']:
            data = [json.loads(line) for line in f]
            data = [{'problem': item['problem'], 'answer': item['answer']} for item in data]
            return data
        else:
            raise ValueError(f"Invalid data type: {data_type}")