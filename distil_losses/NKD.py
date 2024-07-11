import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

import os
import logging

def nkd_loss(logits_s, logits_t, gamma=0.25, temperature=1):
    '''
    logits: batch * seq_len * vocab_size
    '''
    labels = logits_t.argmax(-1, keepdim=True)

    # B * N * class
    B, N, vocab_size = logits_s.shape
    prob_s = F.log_softmax(logits_s, dim=-1)
    prob_t = F.softmax(logits_t, dim=-1)
    # B * N * 1
    prob_gt_s = prob_s.gather(-1, labels)
    prob_gt_t = prob_t.gather(-1, labels).detach()

    loss_t = - (prob_gt_t * prob_gt_s).sum(-1, keepdim=True)

    mask = torch.ones_like(logits_s).scatter_(-1, labels, 0).bool()
    logits_s_mask = logits_s[mask].reshape(B, N, -1)
    logits_t_mask = logits_t[mask].reshape(B, N, -1)
    
    # B * N * class
    prob_other_s = F.log_softmax(logits_s_mask / temperature, dim=-1)
    prob_other_t = F.softmax(logits_t_mask / temperature, dim=-1)     

    loss_non =  (prob_other_t * prob_other_s).sum(-1, keepdim=True)
    # loss_non = - gamma * (temperature**2) * loss_non
    loss_non = - gamma * loss_non

    return loss_t + loss_non 
