import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def bild_loss(logits_s, logits_t, top_k=8, temperature=3, student_led=False):
    """
    Bi-directional Logits Difference loss.

    Args:
        logits_s (torch.Tensor): the student logits, shape (batch_size, seq_len, vocab_size).
        logits_t (torch.Tensor): the teacher logits, shape (batch_size, seq_len, vocab_size).
        top_k (int, optional): choose top-k logits for calculating loss, defaults to 8.
        temperature (int, optional): the temperature, defaults to 3.
        student_led (bool, optional): if true, calculate student-led logits difference loss (t-LD), else t-LD.
    """
    pair_num = top_k * (top_k-1) // 2

    if not student_led:
        # select top-k teacher logits & corresponding student logits
        with torch.no_grad():
            select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
        select_logits_s = torch.gather(logits_s, 2, select_pos)
    else:
        # select top-k student logits & corresponding teacher logits
        select_logits_s, select_pos = torch.topk(logits_s, k=top_k, dim=-1)
        with torch.no_grad():
            select_logits_t = torch.gather(logits_t, 2, select_pos)

    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature

    # calculate logit difference
    def get_prob_diff(logits):
        b, n, v = logits.size()
        i, j = torch.triu_indices(v, v, offset=1)

        logits_diff = logits[..., i] - logits[..., j]

        return logits_diff

    logits_diff_t = get_prob_diff(scaled_logits_t)
    logits_diff_s = get_prob_diff(scaled_logits_s)

    logits_diff_t = F.softmax(logits_diff_t, dim=-1)

    loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t, reduction='none')

    loss = loss.sum(-1, keepdim=True)

    return loss