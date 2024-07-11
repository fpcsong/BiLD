import sys 
sys.path.append("/mnt/data/liminchong/BiLD") 
import deepspeed
from config import *
from utils import *
from prompter import Prompter
from tokenize_functions import *
from distillation import Distillation
from compressalign import CompressAlign
from benchmarking.evaluation import evaluation_func
from ds_utils import get_train_ds_config
from trainers.custom_trainer import *
from callbacks import PeftCallback
from compression.params_reuse import CompressedBloom
from train import log2file

def top1_acc(fp_results, test_results):
    assert len(fp_results) == len(test_results)
    total_input_token_num = 0
    total_input_token_acc = 0
    total_output_token_num = 0
    total_output_token_acc = 0
    for idx, fp_item in enumerate(fp_results):
        test_item = test_results[idx]
        assert fp_item['id'] == test_item['id']
        total_input_token_num += len(fp_item['input_ids'])
        total_output_token_num += len(fp_item['output_ids'])

        fp_input_pred = [item[0][0] for item in fp_item['input_logits']]
        test_input_pred = [item[0][0] for item in test_item['input_logits']]
        total_input_token_acc += (np.array(fp_input_pred) == np.array(test_input_pred)).sum()

        fp_output_pred = [item[0][0] for item in fp_item['output_logits']]
        test_output_pred = [item[0][0] for item in test_item['output_logits']]
        total_output_token_acc += (np.array(fp_output_pred) == np.array(test_output_pred)).sum()
    return total_input_token_acc/total_input_token_num * 100, total_output_token_acc/total_output_token_num * 100


def topk_overlap(fp_results, test_results, k=32):
    assert len(fp_results) == len(test_results)
    assert k <= len(fp_results[0]['input_logits'][0])

    total_input_logits_num = 0
    total_input_logit_overlap = 0
    total_output_logits_num = 0
    total_output_logits_overlap = 0

    for idx, fp_item in enumerate(fp_results):
        test_item = test_results[idx]
        total_input_logits_num += len(fp_item['input_ids']) * k
        total_output_logits_num += len(fp_item['output_ids']) * k

        input_overlap = 0
        for token_idx in range(len(fp_item['input_ids'])):
            fp_input_topk_logits = [item[0] for item in fp_item['input_logits'][token_idx][:k]]
            test_input_topk_logits = [item[0] for item in test_item['input_logits'][token_idx][:k]]
            input_overlap += len(list(set(fp_input_topk_logits) & set(test_input_topk_logits)))
        total_input_logit_overlap += input_overlap
        output_overlap = 0
        for token_idx in range(len(fp_item['output_ids'])):
            fp_output_topk_logits = [item[0] for item in fp_item['output_logits'][token_idx][:k]]
            test_output_topk_logits = [item[0] for item in test_item['output_logits'][token_idx][:k]]
            output_overlap += len(list(set(fp_output_topk_logits) & set(test_output_topk_logits)))
        total_output_logits_overlap += output_overlap

    return total_input_logit_overlap / total_input_logits_num * 100, total_output_logits_overlap / total_output_logits_num * 100

# input_acc, output_acc = top1_acc(fp_results, test_results)
# print('top1 acc', output_acc)

# for k in [8, 16, 32]:
#     input_overlap, output_overlap = topk_overlap(fp_results, test_results, k=k)
#     print('top {} overlap'.format(k), output_overlap)


def load_model_and_tokenizer(args):
    if args.save_teacher_data:
        model_path = args.teacher_model_path
    else:
        model_path = args.student_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=True,
                                                torch_dtype=torch.float16 if args.fp16 else torch.bfloat16)

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ds_config = {
        "replace_with_kernel_inject": True,
        "tensor_parallel": {
            "enabled": True,
            "tp_size": world_size
        },
    }

    if args.deepspeed:
        ds_engine = deepspeed.init_inference(
                                    model,
                                    dtype=torch.half,
                                    config=ds_config,
                                    )
        model = ds_engine.module


    if args.tokenizer_name == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                max_length=CONFIG['max_len'],
                                                pad_token='<|endoftext|>',
                                                eos_token='<|endoftext|>',
                                                padding_side='left',
                                                trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                max_length=CONFIG['max_len'],
                                                padding_side="left",
                                                truncation_side="left",
                                                trust_remote_code=True,
                                                use_fast=True)

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    if args.tokenizer_name in ['llama', 'baichuan', 'cpm']:
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 0
        tokenizer.unk_token_id = 0
    if args.tokenizer_name == 'chatglm':
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 2
        tokenizer.unk_token_id = 0
    if args.tokenizer_name == 'bloom':
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 3
        tokenizer.unk_token_id = 0
    tokenizer.add_special_tokens = False

    return model, tokenizer 


def load_eval_dataset(args):
    val_file_names = args.val_data
    val_file_names = glob.glob(args.val_data+args.val_files_pattern, recursive=True)

    eval_dataset = load_dataset("json", 
                                data_files=val_file_names,
                                split='train',
                                streaming=args.streaming,
                                )
    sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size=args.eval_batch_size,
                                sampler=sampler)
    return eval_dataloader


def save_teacher_data(args):
    teacher_data = []

    # teacher_path = '/mnt/liminchong/results/en_results/teacher/Qwen1.5-0.5B_all/checkpoint-10496'

    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()

    eval_dataloader = load_eval_dataset(args)
    
    for examples in tqdm(eval_dataloader, desc='evaluating', disable=args.local_rank not in [0, -1]):

        examples_tokenized = tokenizer(examples["input"], padding=True, truncation=True, return_tensors='pt')
        input_ids = examples_tokenized["input_ids"]
        attention_mask = examples_tokenized["attention_mask"]

        if torch.cuda.device_count() > 0:
            input_ids = input_ids.to(torch.cuda.current_device())
            attention_mask = attention_mask.to(torch.cuda.current_device())

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_scores=True, 
                                output_logits=True,
                                return_dict_in_generate=True,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False,
                                repetition_penalty=1.2,
                                min_new_tokens=1,
                                max_new_tokens=1024)
            
        sequences = outputs.sequences # tuple, len=1, sub element: torch.Size([51])
        logits = torch.cat(outputs.logits, dim=0) # tuple, len=8 (answer length), sub element: torch.Size([1, 151936])

        # print(f'sequences len: {len(sequences)}\n')
        # print(f'logits len: {len(logits)}\n')
        # print(f'sequences[0].shape: {sequences[0].shape}\n')
        # print(f'logits.shape: {logits.shape}\n')

        for i in range(sequences.size(0)):
            cur_logits_topk, cur_topk_pos = torch.topk(logits, k=args.save_logits_len, dim=-1)

            cur_teacher_data = {
                "task_name": examples["task_name"][i],
                "full_answer": sequences[i].cpu(),
                "logits": cur_logits_topk.cpu(),
                "logits_position": cur_topk_pos.cpu()
            }
            # print(f'cur_logits_topk.shape: {cur_logits_topk.shape}\n')
            # print(f'cur_topk_pos.shape: {cur_topk_pos.shape}\n')
            # exit()
            teacher_data.append(cur_teacher_data)

    torch.save(teacher_data, os.path.join(args.teacher_save_path, f'{args.model_name}.pt'))

    return


def count_common_elements(A: torch.tensor, B: torch.tensor):
    from collections import Counter

    # 将 tensor 转换为 list
    A_list = A.tolist()
    B_list = B.tolist()
    # 统计每个列表中元素出现的次数
    A_count = Counter(A_list)
    B_count = Counter(B_list)
    # 计算相同元素的总数
    common_count = 0
    for elem in A_count:
        if elem in B_count:
            common_count += min(A_count[elem], B_count[elem])                
    return common_count


def eval_acc_overlap(args):
    # teacher_data = torch.load(os.path.join(args.teacher_save_path, f'{args.model_name}.pt'))
    top_1_acc = 0
    top_k_overlap = 0

    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()

    teacher_data = torch.load(os.path.join(args.teacher_save_path, f'{args.model_name}.pt'))
    teacher_data = teacher_data

    print(f"successfully load {os.path.join(args.teacher_save_path, f'{args.model_name}.pt')}")

    data_num = float(len(teacher_data))

    print(f'data_num: {data_num}')

    for teacher_examples in tqdm(teacher_data, desc='evaluating', disable=args.local_rank not in [0, -1]):

        answer_len = teacher_examples["logits_position"].shape[0]

        # print(f'answer_len: {answer_len}')
        # print(f'task_name: {teacher_examples["task_name"]}')
        # print(f'full_answer len: {teacher_examples["full_answer"].shape}')
        # print(f'logits shape: {teacher_examples["logits"].shape}')
        # print(f'logits pos shape: {teacher_examples["logits_position"].shape}')

        input_ids = teacher_examples["full_answer"].unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        if torch.cuda.device_count() > 0:
            input_ids = input_ids.to(torch.cuda.current_device())
            attention_mask = attention_mask.to(torch.cuda.current_device())

        with torch.no_grad():    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # print(f'raw logits shape: {outputs.logits.shape}')
        
        logits = outputs.logits.squeeze(0)
        logits = logits[-(answer_len + 1):-1, :]

        # print(f'logits shape: {logits.shape}')

        pos_t = teacher_examples["logits_position"].to(torch.cuda.current_device())

        for i in range(args.eval_batch_size):
            logits_s, pos_s = torch.topk(logits, k=args.overlap_topk, dim=-1)
            
            # calulate top-1 acc
            top_1_pos_s = pos_s[:, 0]

            # print(f'top_1_pos_s.shape: {top_1_pos_s.shape}')
            # print(f'top_1_pos_s: {top_1_pos_s}')
            # print(f'full_answer: {teacher_examples["full_answer"]}')
            # print('#######################')
            # print(f'top_1_pos_s, skip: {" ".join(tokenizer.batch_decode(top_1_pos_s, skip_special_tokens=True))}')
            # print('#######################')
            # print(f'top_1_pos_s, no skip: {" ".join(tokenizer.batch_decode(top_1_pos_s, skip_special_tokens=False))}')
            # print('#######################')
            # print(f'full_answer, skip: {" ".join(tokenizer.batch_decode(teacher_examples["full_answer"], skip_special_tokens=True))}')
            # print('#######################')
            # print(f'full_answer, no skip: {" ".join(tokenizer.batch_decode(teacher_examples["full_answer"], skip_special_tokens=False))}')
            # print('#######################')
            # print('\n\n')

            top_1_pos_t = pos_t[:, 0]
            equal_top_1 = (top_1_pos_s == top_1_pos_t)
            equal_top_1_count = torch.sum(equal_top_1).item()
            top_1_acc += equal_top_1_count / float(answer_len)

            # print(f'top_1_pos_s: {top_1_pos_s}')
            # print(f'top_1_pos_t: {top_1_pos_t}')
            # print(f'single top-1 acc: {equal_top_1_count / float(answer_len)}')

            # calculate top-k overlap
            top_k_pos_s = pos_s[:, :args.overlap_topk]
            top_k_pos_t = pos_t[:, :args.overlap_topk]
            
            overlap_ratio = 0

            for j in range(answer_len):
                top_k_pos_s_unique = top_k_pos_s[j, :]
                top_k_pos_t_unique = top_k_pos_t[j, :]
                overlap_count = count_common_elements(top_k_pos_s_unique, top_k_pos_t_unique)
                overlap_ratio += (overlap_count / float(args.overlap_topk))

            top_k_overlap += (overlap_ratio / float(answer_len))

            # print(f'top_k_pos_s: {top_k_pos_s}')
            # print(f'top_k_pos_t: {top_k_pos_t}')
            # print(f'single top_k_overlap: {overlap_ratio}')

    top_1_acc /= data_num
    top_k_overlap /= data_num

    top_1_acc *= 100
    top_k_overlap *= 100

    print(f'top-1 acc: {top_1_acc}')
    print(f'top-{args.overlap_topk} overlap: {top_k_overlap}')

    log2file(args, f'top-1 acc: {top_1_acc}\ntop-{args.overlap_topk} overlap: {top_k_overlap}')

    return top_1_acc, top_k_overlap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate top-1 acc and top-p overlap')
    parser.add_argument('-output_dir', '--output_dir', type=str, \
                        default=CONFIG['output_dir'], help='output_dir')
    parser.add_argument('-acc_step',
                        '--accumulation_steps',
                        default=CONFIG['accumulation_steps'],
                        type=int,
                        required=False)
    parser.add_argument('-epochs', '--epochs', type=int, \
                        default=CONFIG['epochs'], help='training epochs')
    parser.add_argument('-max_steps', '--max_steps', type=int, \
                        default=CONFIG['max_steps'], help='training max_steps')
    parser.add_argument('-max_len', '--max_len', type=int, \
                        default=CONFIG['max_len'], help='training max_len')
    parser.add_argument('-save_steps', '--save_steps', type=int, \
                        default=CONFIG['save_steps'], help='save_steps')
    parser.add_argument('-eval_steps', '--eval_steps', type=int, \
                        default=CONFIG['eval_steps'], help='eval_steps')
    parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, \
                        default=64, help='max_new_tokens')
    parser.add_argument('-num_beams', '--num_beams', type=int, \
                        default=CONFIG['num_beams'], help='num_beams')
    parser.add_argument('-lora_r', '--lora_r', type=int, \
                        default=LORA_CONFIG['r'], help='lora r')
    parser.add_argument('-lr', '--learning_rate', type=float, \
                        default=CONFIG['learning_rate'], help='learning rate')
    parser.add_argument('-alpha', '--alpha', type=float, \
                        default=CONFIG['alpha'], help='weight of distillation loss')
    parser.add_argument('-temperature', '--temperature', type=float, \
                        default=CONFIG['temperature'], help='temperature for CE distillation loss')
    parser.add_argument('-v_data','--val_data', type=str, \
                        default=CONFIG['val_data'], help='the data used for evaluation')
    parser.add_argument('-t_data','--train_data', type=str, \
                        default=CONFIG['train_data'], help='the data used for instructing tuning')
    parser.add_argument('-p_data', '--pretrain_data', type=str, \
                        default=CONFIG['pretrain_data'], help='the data used for pretraining')
    parser.add_argument('--local_rank', default=-1, type=int,\
                        help='node rank for distributed training')
    parser.add_argument('--master_port', default="29501", type=str,\
                        help='master_port')
    parser.add_argument('--model_name', type=str, required=True,\
                        default=CONFIG['model_name'], help='the name of target llm model')
    parser.add_argument('--tokenizer_name', type=str, required=False,\
                        default='', help='the name of target llm tokenizer')
    parser.add_argument('--teacher_model_path', type=str, required=False,\
                        default=CONFIG['model_path'], help='the folder contains model weights')
    parser.add_argument('--student_model_path', type=str, required=False,\
                        default=CONFIG['student_model_path'], help='the folder contains student model weights')
    parser.add_argument('--template_name', type=str, \
                        default='alpaca_short', help='instruct template')
    parser.add_argument('--loss_type', type=str, \
                        default=CONFIG['loss_type'], help='loss type')
    parser.add_argument('--deepspeed', type=str, \
                        default=CONFIG['deepspeed_config'], help='deepspeed config file path')
    parser.add_argument('-fp16', '--fp16', action='store_true', \
                        default=False, help='use fp16')
    parser.add_argument('-streaming', '--streaming', action='store_true',default=False)
    parser.add_argument('--train_files_pattern', '-train_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('--val_files_pattern', '-val_files_pattern', type=str, default='//*.jsonl')
    parser.add_argument('-output_file', '--output_file',default="output.log")
    parser.add_argument('-save_teacher_data', '--save_teacher_data', action='store_true', \
                        default=False, help='save teacher logits')
    parser.add_argument('-teacher_save_path', '--teacher_save_path', type=str, \
                        help='the path to save teacher logits')
    parser.add_argument('-save_logits_len', '--save_logits_len', type=int, default=1024)
    parser.add_argument('-overlap_topk', '--overlap_topk', type=int, default=32)
    parser.add_argument('-num_workers', '--num_workers', type=int, \
                        default=12, help='number of workers')

    set_random_seed(42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    args.output_file = os.path.join(args.output_dir, "logits_eval.txt")
    if args.tokenizer_name == '':
        args.tokenizer_name = args.model_name
    args_dict = vars(args)
    for k, v in args_dict.items():
        CONFIG[k] = v
    LORA_CONFIG['r'] = args.lora_r
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    CONFIG['world_size'] = world_size
    if args.deepspeed:
        deepspeed.init_distributed("nccl")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.local_rank in [0, -1]:
        print(CONFIG)
    datasets.config.IN_MEMORY_MAX_SIZE = 128 * 1024 * 1024

    if args.save_teacher_data:
        save_teacher_data(args)
    else:
        eval_acc_overlap(args)
