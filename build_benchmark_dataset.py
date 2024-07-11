# -- coding: utf-8 --**

'''
'''
import torch
import datasets
from datasets import load_dataset
import argparse
import os
from prompter import Prompter
    
def main(args):
    tasks = os.listdir(args.path)
    for task in tasks:
        if os.path.isfile(os.path.join(args.path,task)):
            continue
        print('processing task {}'.format(task))
        prompter = Prompter(task)
        # 转成 instruction, input, output, task_name 格式
        train_data_file = os.path.join(args.path, task, 'train/formated*.jsonl')
        eval_data_file = os.path.join(args.path, task, 'eval/formated*.jsonl')
        trainset = load_dataset("json", data_files=train_data_file, split='train')
        evalset = load_dataset("json", data_files=eval_data_file, split='train')
        func = lambda item: {
            'instruction' : "",
            'input' : prompter.generate_prompt(item),
            'output' : item.get(prompter.template['label_key']),
            'response_split': prompter.template['response_split']
        }
        remove_features = list(trainset.features.keys())
        print('origin features in {} are'.format(task))
        print(remove_features)
        for feat in ['instruction', 'input', 'output', 'task_name', 'response_split']:
            if feat in remove_features:
                remove_features.remove(feat)
        # 采样
        if args.num_samples_per_task != -1:
            trainset = trainset.shuffle().train_test_split(
                min(len(trainset)-1, args.num_samples_per_task)
            )['test']
            evalset = evalset.shuffle().train_test_split(
                min(len(evalset)-1, 2048)
            )['test']

        trainset = trainset.map(func, remove_columns=remove_features)
        trainset.to_json(
            os.path.join(args.output_path, task, 'train', task+'.jsonl'),
            batch_size=2048, num_proc=32, lines=True, force_ascii=False
        )
        evalset = evalset.map(func, remove_columns=remove_features)
        evalset.to_json(
            os.path.join(args.output_path, task, 'eval', task+'.jsonl'),
            batch_size=2048, num_proc=32, lines=True, force_ascii=False
        )


if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='build a dataset for evaluation pretrained language models')
    parser.add_argument('--num_samples_per_task', '-num_samples_per_task', type=int, default=256, required=False)
    parser.add_argument('--output_path', '-output_path', type=str, default='./benchmarking/mix/', required=False)
    parser.add_argument('--path', '-path', type=str, default='./benchmarking/datasets', required=False)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
