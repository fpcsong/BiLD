import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
def bild_loss(logits_s, logits_t, top_k=8, temperature=3, student_led=False):
    """
    Bi-directional Logits Difference loss.
    """
    pair_num = top_k * (top_k-1) // 2

    if not student_led:
        with torch.no_grad():
            select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
        select_logits_s = torch.gather(logits_s, 2, select_pos)
    else:
        select_logits_s, select_pos = torch.topk(logits_s, k=top_k, dim=-1)
        with torch.no_grad():
            select_logits_t = torch.gather(logits_t, 2, select_pos)

    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature

    def get_prob_diff(logits):
        b, n, v = logits.size()
        i, j = torch.triu_indices(v, v, offset=1, device=logits.device)
        logits_diff = logits[..., i] - logits[..., j]
        return logits_diff

    logits_diff_t = get_prob_diff(scaled_logits_t)
    logits_diff_s = get_prob_diff(scaled_logits_s)

    logits_diff_t = F.softmax(logits_diff_t, dim=-1)
    loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t, reduction='none')
    loss = loss.sum(-1, keepdim=True)
    
    return loss


def vanilla_kl_loss(logits_s, logits_t, temperature=1):
    scaled_logits_t = logits_t / temperature
    scaled_logits_s = logits_s / temperature
    
    p_T = F.softmax(scaled_logits_t, dim=-1)
    loss = F.kl_div(F.log_softmax(scaled_logits_s, dim=-1), p_T, reduction='none')
    loss = loss.sum(-1, keepdim=True)
    
    return loss


# Hyperparameters
batch_size = 2
seq_len = 2048
vocab_size = 150000

# Setup micro batch size
micro_batch_size = 1

# Random sample generation

# Measure bild_loss time
logits_s = torch.randn(batch_size, seq_len, vocab_size)
logits_t = torch.randn(batch_size, seq_len, vocab_size)
logits_s = logits_s.cuda()
logits_t = logits_t.cuda()
print('logits shape', logits_s.shape, '1000 times')
for k in (8, 16, 32, 64, 128):
    bild_loss_time = 0
    for i in tqdm(range(1000)):
        start_time = time.time()
        _ = bild_loss(logits_s, logits_t, top_k=k)
        torch.cuda.synchronize()  # Wait for GPU to finish calculations
        bild_loss_time += time.time() - start_time
    print("topk={}, bild_loss time: {:0.4f} seconds".format(k, bild_loss_time))
    

# Measure vanilla_kl_loss time
vanilla_kl_loss_time = 0
for i in tqdm(range(1000)):
    start_time = time.time()
    _ = vanilla_kl_loss(logits_s, logits_t)
    torch.cuda.synchronize()  # Wait for GPU to finish calculations
    vanilla_kl_loss_time += time.time() - start_time

print("vanilla_kl_loss time: {:0.4f} seconds".format(vanilla_kl_loss_time))