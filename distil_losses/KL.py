import torch
import torch.nn as nn
import torch.nn.functional as F

def vanilla_kl_loss(logits_s, logits_t, temperature=1):

    scaled_logits_t = logits_t / temperature
    scaled_logits_s = logits_s / temperature
    
    p_T = F.softmax(scaled_logits_t, dim=-1)
    loss = F.kl_div(F.log_softmax(scaled_logits_s, dim=-1), p_T, reduction='none')
    loss = loss.sum(-1, keepdim=True)

    return loss


def top_kl_loss(logits_s, logits_t, temperature=3, top_k=1024):

    # select top-k teacher logits & corresponding student logits
    with torch.no_grad():
        select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
    select_logits_s = torch.gather(logits_s, 2, select_pos)


    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature
    
    p_T = F.softmax(scaled_logits_t, dim=-1)
    loss = F.kl_div(F.log_softmax(scaled_logits_s, dim=-1), p_T, reduction='none')
    loss = loss.sum(-1, keepdim=True)

    return loss


def reverse_kl_loss(logits_s, logits_t, temperature=1):
    scaled_logits_t = logits_t / temperature
    scaled_logits_s = logits_s / temperature

    prob_t_log = F.log_softmax(scaled_logits_t, dim=-1, dtype=torch.float32)
    prob_s = F.softmax(scaled_logits_s, dim=-1, dtype=torch.float32)
    
    loss = F.kl_div(prob_t_log, prob_s, reduction='none')

    loss = loss.sum(-1, keepdim=True)

    return loss
