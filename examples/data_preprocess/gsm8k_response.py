# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset with deepseek responses to parquet format
"""

import re
import os
import datasets
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse


# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split('#### ')[1].replace(',', '')
#     return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='/home/zhaoyiz/projects/reasoning/efficient-reasoning/outputs/openai_gsm8k_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json')
    parser.add_argument('--test_data_path', default='/home/zhaoyiz/projects/reasoning/efficient-reasoning/outputs/openai_gsm8k_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    with open(args.train_data_path, 'r') as f:
        train_dataset = json.load(f)
    with open(args.test_data_path, 'r') as f:
        test_dataset = json.load(f)
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']

    prompt_template = "You will be given the student's response. Read through the student's response step by step to see if there are mistakes. Answer True if there are no mistakes, and False otherwise. Put your final answer (True or False) within \\boxed{{}}. \n\nResponse: {response}"

    #!TODO
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            # question_raw = example.pop('question')

            # question = question_raw + ' ' + instruction_following

            # answer_raw = example.pop('answer')
            # solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
