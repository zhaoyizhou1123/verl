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
    parser.add_argument('--train_data_path', default='/home/zhaoyiz/projects/reasoning/efficient-reasoning/outputs/openai_gsm8k_train_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json')
    parser.add_argument('--test_data_path', default='/home/zhaoyiz/projects/reasoning/efficient-reasoning/outputs/openai_gsm8k_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B_32768.json')
    parser.add_argument('--local_dir', type=str, default="~/data/gsm8k_verify_deepseek")
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    with open(args.train_data_path, 'r') as f:
        train_dataset = json.load(f)
    with open(args.test_data_path, 'r') as f:
        test_dataset = json.load(f)
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']
    train_dataset = datasets.Dataset.from_list(train_dataset)
    test_dataset = datasets.Dataset.from_list(test_dataset)

    prompt_template = "You are going to verify a student's answer. You will be given a math question, as well as the student's response. The response contains the reasoning process as well as the final answer. First extract the student's final answer, then read through the student's response to see if there are mistakes. Finally, check the student's answer. Your final answer is True if the student's answer is true, and False otherwise. Put your final answer (True or False) within \\boxed{{}}. \n\nQuestion: {question}\n\nResponse: {response}"

    #!TODO
    # add a row to each data item that represents a unique id
    def process_fn(example, idx):
        # question_raw = example.pop('question')

        # question = question_raw + ' ' + instruction_following

        # answer_raw = example.pop('answer')
        # solution = extract_solution(answer_raw)
        data = {
            "data_source": "verify",
            "prompt": [{
                "role": "user",
                "content": prompt_template.format(
                    question = example['question'],
                    response = example['responses'][0]
                    )}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(example["accuracy"][0])
            },
            "extra_info": {
                "extracted_answer": example["prediction"],
                "ground_truth": example["gold"]
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn, with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True, remove_columns=test_dataset.column_names )

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
