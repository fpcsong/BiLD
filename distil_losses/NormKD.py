import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def normkd_loss(logits_s, logits_t, T_norm=1.5):
    """
    NormKD Loss。
    """
    sigma_t = torch.std(logits_t, -1, keepdim=True)
    sigma_s = torch.std(logits_s, -1, keepdim=True)   
    
    T_t = sigma_t * T_norm
    T_s = sigma_s * T_norm

    scaled_logits_t = logits_t / T_t
    scaled_logits_s = logits_s / T_s

    scaled_logits_t = F.softmax(scaled_logits_t, dim=-1)

    loss = F.kl_div(F.log_softmax(scaled_logits_s, dim=-1), scaled_logits_t, reduction='none') 
    loss = loss.sum(-1, keepdim=True) * (T_t ** 2)

    return loss