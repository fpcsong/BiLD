import os
import json
import glob
import random
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from prompter import Prompter

datasets_dir='./benchmarking/datasets'

task_list = os.listdir(datasets_dir)
print(f'task list: {task_list}')

task_list = ['arc-c', 'arc-e', 'boolq', 'cb', 'copa', 'hellaswag', 'multirc', 'piqa', 'record', 'rte', 'wic', 'winogrande', 'wsc']

# delete generated data
for task in task_list:
    if os.path.exists(os.path.join(datasets_dir, task, f'train/formated_train_{task}.jsonl')):
        os.remove(os.path.join(datasets_dir, task, f'train/formated_train_{task}.jsonl'))
        os.remove(os.path.join(datasets_dir, task, f'eval/formated_eval_{task}.jsonl'))
    if os.path.exists(os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl')):
        os.remove(os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl'))
        os.remove(os.path.join(datasets_dir, task, f'eval/input_eval_{task}.jsonl'))

# format the raw data then save as local file (formated_*.jsonl)
for task in task_list:
    if task in ['reformat.py']:
        continue
    if os.path.exists(os.path.join(datasets_dir, task, f'train/formated_train_{task}.jsonl')):
        print('task {} has been processed. No more action will be done.'.format(task))
        continue
    print('\n\n\n')
    print('processing task {}'.format(task))
    train_files = glob.glob(os.path.join(datasets_dir, task+'/train/*.json*'), recursive=True)
    print("train set data files", train_files)
    eval_files = glob.glob(os.path.join(datasets_dir, task+'/eval/*.json*'), recursive=True)
    print("eval set data files", eval_files)

    if task == 'arc-c':
        def proc_func(item):
            for i in range(len(item['question']['choices'])):
                if item['question']['choices'][i]['label'] == item['answerKey']:
                    return item['question']['choices'][i]['text']
            return None
        func = lambda item: {
            'task_name': 'arc-c',
            'input_question': item['question']['stem'],
            'options': [item['question']['choices'][i]['text'] for i in range(len(item['question']['choices']))],
            'answer': proc_func(item)
        }
        remove_columns = ['id', 'question', 'answerKey']

    if task == 'arc-e':
        def proc_func(item):
            for i in range(len(item['question']['choices'])):
                if item['question']['choices'][i]['label'] == item['answerKey']:
                    return item['question']['choices'][i]['text']
            return None
        func = lambda item: {
            'task_name': 'arc-e',
            'input_question': item['question']['stem'],
            'options': [item['question']['choices'][i]['text'] for i in range(len(item['question']['choices']))],
            'answer': proc_func(item)
        }
        remove_columns = ['id', 'question', 'answerKey']

    if task == 'boolq':
        func = lambda item: {
            'task_name': 'boolq',
            'labelKey': 'true' if item['label'] == True else 'false'
        }
        remove_columns = ['idx', 'label']

    if task == 'cb':
        func = lambda item: {
            'task_name': 'cb'
        }
        remove_columns = ['idx']

    if task == 'copa':
        func = lambda item: {
            'task_name': 'copa',
            'answer': item['choice1'] if item['label'] == 0 else item['choice2']
        }
        remove_columns = ['label', 'idx']

    if task == 'hellaswag':
        func = lambda item: {
            'task_name': 'hellaswag',
            'passage': item['ctx'],
            'answer': item['endings'][item['label']]
        }
        remove_columns = ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'split', 'split_type', 'label', 'source_id']

    if task == 'rte':
        func = lambda item: {
            'task_name': 'rte'
        }
        remove_columns = ['idx']

    if task == 'wic':
        func = lambda item: {
            'task_name': 'wic',
            'labelKey': 'true' if item['label'] == True else 'false' 
        }
        remove_columns = ['idx', 'label', 'start1', 'start2', 'end1', 'end2', 'version']

    if task == 'winogrande':
        func = lambda item: {
            'task_name': 'winogrande',
            'answerKey': item['option1'] if int(item['answer']) == 1 else item['option2']
        }
        remove_columns = ['qID', 'answer']

    if task == 'wsc':
        func = lambda item: {
            'task_name': 'wsc',
            'span1_text': item['target']['span1_text'],
            'span2_text': item['target']['span2_text'],
            'labelKey': 'true' if item['label'] == True else 'false' 
        }
        remove_columns = ['target', 'idx', 'label']


    if task in ['arc-c', 'arc-e', 'boolq', 'cb', 'copa', 'hellaswag', 'rte', 'wic', 'winogrande', 'wsc']:
        # process train set
        trainset = load_dataset("json", 
                                data_files=train_files,
                                split='train')
        trainset = trainset.map(
            func,
            remove_columns=remove_columns
        )
        trainset.to_json(os.path.join(datasets_dir, task, 'train/formated_train_{}.jsonl'.format(task)),
                            lines=True, force_ascii=False)
        
        # process eval set
        evalset = load_dataset("json", 
                                data_files=eval_files,
                                split='train')
        evalset = evalset.map(
            func,
            remove_columns=remove_columns
        )
        evalset.to_json(os.path.join(datasets_dir, task, 'eval/formated_eval_{}.jsonl'.format(task)),
                            lines=True, force_ascii=False)
        
    elif task in ['multirc', 'piqa', 'record']: # for data in different format (e.g. store label in other file), we process them seperately
        def process_multirc(jsonset, mode='train'):
            processed_data = []
            for item in jsonset:
                text = item['passage']['text']
                questions = item['passage']['questions']
                for question in questions:
                    q_text = question['question']
                    answers = question['answers']
                    choices_list = []
                    answers_list = []
                    for answer in answers:
                        a_text = answer['text']
                        choices_list.append(a_text)
                        if answer['label'] == 1:
                            answers_list.append(a_text)
                    
                    if mode == 'train':
                        answers_reformat = '\n'.join(answers_list)
                    elif mode == 'eval':
                        answers_reformat = '★'.join(answers_list) # for the convenience of rebuild on eval step
                    
                    if answers_reformat == '':
                        answers_reformat = 'none'

                    processed_data.append({'text': text, 'question': q_text, 'choices': choices_list, 'answers': answers_reformat, 'task_name': 'multirc'})

            return processed_data

        def process_piqa(jsonset, mode='train'):
            processed_data = []

            label_path = os.path.join(datasets_dir, task, 'train', 'train-labels.lst') if mode == 'train' \
                else os.path.join(datasets_dir, task, 'eval', 'valid-labels.lst')

            with open(label_path, 'r') as file:
                lines = file.readlines()
                for line, item in zip(lines, jsonset):
                    label = int(line.strip())
                    answer = item['sol1'] if label == 0 else item['sol2']
                    processed_data.append({'goal': item['goal'], 'sol1': item['sol1'], 'sol2': item['sol2'], 'answer': answer, 'task_name': 'piqa'})

            return processed_data

        def process_record(jsonset, mode='train'): # datapoint from ReCoRD may have multiple correct answers 
            processed_data = []

            for item in jsonset:
                for i in range(len(item['qas'])):
                    possible_answers = set()
                    for answer in item['qas'][i]['answers']:
                        possible_answers.add(answer['text'])
                    possible_answers = list(possible_answers)
                    if mode == 'train':
                        '''
                        As ReCoRD dataset may have multiple correct answers for one datapoint,
                        we randomly choose one answer for training. 
                        '''
                        random_chosed_answer = random.choice(possible_answers)
                        processed_data.append({'passage': item['passage']['text'], 'query': item['qas'][i]['query'], 
                                            'answer': random_chosed_answer, 'taskname': 'record'})
                    elif mode == 'eval':
                        '''
                        Keep all labels for testing. 
                        We use '★' to combine the list elements and rebuild the list when evaluating. 
                        '''
                        possible_answers_string = '★'.join(possible_answers)
                        processed_data.append({'passage': item['passage']['text'], 'query': item['qas'][i]['query'], 
                                            'answer': possible_answers_string, 'taskname': 'record'})

            return processed_data
        
        process_func_sep = {
            'multirc': process_multirc, 
            'piqa': process_piqa, 
            'record': process_record
        }.get(task)

        # process train set
        trainset = load_dataset("json", 
                                data_files=train_files,
                                split='train')
        
        trainset = process_func_sep(trainset, mode='train')

        with open(os.path.join(datasets_dir, task, 'train/formated_train_{}.jsonl'.format(task)), 'w') as f:
            for entry in tqdm(trainset):
                json.dump(entry, f)
                f.write('\n')
        
        # process eval set
        evalset = load_dataset("json", 
                                data_files=eval_files,
                                split='train')
        
        evalset = process_func_sep(evalset, mode='eval')

        with open(os.path.join(datasets_dir, task, 'eval/formated_eval_{}.jsonl'.format(task)), 'w') as f:
            for entry in tqdm(evalset):
                json.dump(entry, f)
                f.write('\n')

    else:
        raise NotImplementedError
    
# convert the formated data to the input format, save as input_*.jsonl
for task in task_list:
    if task in ['reformat.py']:
        continue
    if os.path.exists(os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl')):
        continue
    print('\n\n\n')
    print('processing task {}, converting to input format'.format(task))
    prompter = Prompter(task)
    # 转成 instruction, input, output, task_name 格式
    train_data_file = os.path.join(datasets_dir, task, 'train/formated*.jsonl')
    eval_data_file = os.path.join(datasets_dir, task, 'eval/formated*.jsonl')
    trainset = load_dataset("json", data_files=train_data_file, split='train')
    evalset = load_dataset("json", data_files=eval_data_file, split='train')
    func = lambda item: {
        'instruction' : "",
        'input' : prompter.generate_prompt(item),
        'output' : item.get(prompter.template['label_key']),
        'task_name': task,
        'response_split': prompter.template['response_split']
    }
    remove_features = list(trainset.features.keys())
    print('origin features in {} are'.format(task))
    print(remove_features)
    for feat in ['instruction', 'input', 'output', 'task_name', 'response_split']:
        if feat in remove_features:
            remove_features.remove(feat)

    trainset = trainset.map(func, remove_columns=remove_features)
    trainset.to_json(
        os.path.join(datasets_dir, task, f'train/input_train_{task}.jsonl'),
        batch_size=2048, num_proc=32, lines=True, force_ascii=False
    )
    evalset = evalset.map(func, remove_columns=remove_features)
    evalset.to_json(
        os.path.join(datasets_dir, task, f'eval/input_eval_{task}.jsonl'),
        batch_size=2048, num_proc=32, lines=True, force_ascii=False
    )

