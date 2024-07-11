# -- coding: utf-8 --**
from config import *

"""
A dedicated helper to manage templates and prompt building.
"""

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def sharpen(logits, T):
    probs = F.softmax(logits / T, dim=-1)
    return probs

def print0(*message):
    """If distributed is initialized, print only on rank 0."""
    if int(os.environ.get('LOCAL_RANK', -1)) in [-1, 0]:
            print(*message, flush=True)

def print_in_rank(message, rank=0):
    """If distributed is initialized, print only on rank {rank}."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            print(rank, message, flush=True)
    else:
        print(message, flush=True)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print0(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def get_trainable_state_dict(model, state_dict):
    new_dict = {}
    state_dict = model.student_model.state_dict()
    for k, param in state_dict.items():
        if 'student' in k:
            new_k = k.split('student_model.')[1]
            new_dict[new_k] = param
        else:
            new_dict[k] = param
    return new_dict

def moving_average(params, ema_params, beta=0.992, device=None, zero_stage=0):
    def _z3_params_to_fetch(param_list):
        return [
            p for p in param_list
            if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
        ]
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        scale = 1e3
        for param, param_ema in zip(params,
                                    ema_params):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(param.data * scale, param_ema.data * scale, beta) / scale)

class Concatenator(object):
    def __init__(self, chunk_size=2048, residual={"input_ids": [], "attention_mask": [], "labels": []}):
        self.chunk_size=chunk_size
        self.residual = residual
        
    def __call__(self, batch):
        item_lens = [len(item) for item in batch['input_ids']]
        groups = []
        group = []
        for idx, item_len in enumerate(item_lens):
            if sum([item_lens[_idx] for _idx in group]) + item_len > self.chunk_size:
                random.shuffle(group)
                groups.append(group)
                group = [idx]
            else:
                group.append(idx)
        if group:
            random.shuffle(group)
            groups.append(group)

        total_length = sum(item_lens)
        result = {
            k: [
                list(chain(*[batch[k][_idx] for _idx in group])) 
                for group in groups
            ] for k, _ in self.residual.items()
        }
        return result
        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        # result["labels"] = result["input_ids"].copy()

        return result

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)