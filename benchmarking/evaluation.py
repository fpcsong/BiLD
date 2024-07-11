from ignite.metrics import Rouge
import logging
import re
import string
from sympy import false
from tqdm import tqdm
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(ground_truth, prediction):
    prediction_tokens = prediction.strip().split()
    ground_truth_tokens = ground_truth.strip().split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluation_func(task, labels, predicts):

    if task in ['c3', 'clozet', 'cmrc2018', 'math23k', 'ocnli', 'sst2', 'wantwords',
                'arc-c', 'arc-e', 'boolq', 'copa', 'hellaswag', 'piqa', 'rte', 'wic', 'winogrande', 'wsc']:
        total = len(labels) * 1.0
        acc = 0
        for gt_label, pred in zip(labels, predicts):
            if gt_label.strip() == pred.strip():
                acc += 1
        return {'task_name': task, 'metric': 'Accuracy', 'result': acc / total}
    
    if task == 'cb':
        total = len(labels) * 1.0
        acc = 0
        f1 = 0
        for gt_label, pred in zip(labels, predicts):
            if gt_label.strip() == pred.strip():
                acc += 1
                f1 += 1
            else:
                f1 += f1_score(gt_label, pred)
        return {'task_name': task, 'metric': 'F1/Accuracy', 'result': [f1 / total, acc / total]}
    
    if task == 'multirc': # multiple correct answer
        total = len(labels) * 1.0
        f1a = 0
        em_score = 0

        for gt_labels, pred in zip(labels, predicts):

            mistake = True
            answers_list =  gt_labels.split('★')
            pred_list = pred.split('\n')

            f1a += f1_score(' '.join(answers_list), ' '.join(pred_list))

            em_score += int(set(pred_list) == set(answers_list))

        return {'task_name': task, 'metric': 'F1a/EM', 'result': [f1a / total, em_score / total]}
    
    if task == 'record': # multiple correct answer
        total = len(labels) * 1.0
        acc = 0
        f1 = 0
        for gt_labels, pred in zip(labels, predicts):

            mistake = True
            answers_list =  gt_labels.split('★')

            for possible_answer in answers_list:
                if possible_answer.strip() == pred.strip():
                    mistake = False
                    acc += 1
                    f1 += 1
                    break

            if mistake: # calculate the marco-f1 over all possible answers
                f1_temp = 0
                for possible_answer in answers_list:
                    f1_temp += f1_score(possible_answer, pred)
                f1_temp /= len(answers_list)
                f1 += f1_temp

        return {'task_name': task, 'metric': 'F1/Accuracy', 'result': [f1 / total, acc / total]}
    
    if task in ['cluener']:
        tp, fp, fn = 0, 0, 0
        for gold, pre in zip(labels, predicts):
            gold_kv = []
            for kvs in gold.split(';'):
                k, vs = kvs.split(':')
                for v in vs.split(','):
                    gold_kv.append(k+v)
            pre_kv = []
            for all_kvs in pre.split(';'):
                kvs = all_kvs.split(':')
                if len(kvs) > 1:
                    vs = kvs[1].split(',')
                    for v in vs:
                        pre_kv.append(kvs[0]+v)
            for item in pre_kv:
                if item in gold_kv:
                    tp += 1
                else:
                    fp += 1
            for item in gold_kv:
                if item not in pre_kv:
                    fn += 1
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        return {'task_name': task, 'metric': 'f1', 'result': f}

    if task == 'lcsts':
        m = Rouge(variants=[1,2,'L'], multiref='best')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
        preds = []
        y = []
        for l, p in zip(labels, predicts):
            y = tokenizer.tokenize(l)
            preds = tokenizer.tokenize(p)
            m.update(([preds], [[y]]))
        res = m.compute()
        rouge_1_F = res['Rouge-1-F']
        rouge_2_F = res['Rouge-2-F']
        rouge_L_F = res['Rouge-L-F']
        # result = (rouge_1_F + rouge_2_F + rouge_L_F) / 3
        result = rouge_1_F

        return {'task_name': 'lcsts', 'metric' : 'rouge', 'result': result}