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

from .helper import DATASET_KEYS, RESPONSE_EXTRACTOR, RESPONSE_COMPARATOR

def compute_score(solution_str, ground_truth, format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    dataset_name = "verify"
    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    eq = RESPONSE_COMPARATOR[dataset_name]
    answer = RESPONSE_EXTRACTOR[dataset_name](solution_str)
    acc = eq(ground_truth, answer)
    if acc:
        return score
    if eq(ground_truth, "True") or eq(ground_truth, "False"):
        return format_score
    
def test():
    solution_str = "abcd"
    ground_truth=False
    compute_score(solution_str, str(ground_truth), "verify")

if __name__ == '__main__':
    test()