import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def get_topk_logits(logits, labels, topk=1024):
    batch_size, seq_length, vocab_size = logits.shape
    # set 0 to -100
    with torch.no_grad():
        fake_logits = logits.reshape(-1, vocab_size).detach().clone()
        fake_labels = labels.reshape(-1)
        for bid in range(fake_logits.shape[0]):
            if fake_labels[bid] > -1:
                fake_logits[bid][fake_labels[bid]] += 1e5
        mask_pos = torch.argsort(fake_logits, dim=-1)
        mask_pos = mask_pos.reshape(logits.shape)
        select_pos = mask_pos[:,:,-topk:]
    select_logits = torch.gather(logits, 2, select_pos)

    select_logits = select_logits.reshape(-1, topk)
    return select_logits, select_pos

def dkd_loss(logits_student, logits_teacher, labels, alpha=0.5, beta=0.5, t_norm=2.0, reduction='none'):

    topk = logits_student.shape[-1]
    logits_t, select_pos = get_topk_logits(logits_teacher, labels, topk=topk)
    logits_s = torch.gather(logits_student, 2, select_pos)
    logits_s = logits_s.reshape(-1, topk)
    labels = torch.ones(topk).long().to(logits_t.device) * (topk-1)
    labels = labels.reshape(-1)
    gt_mask = torch.zeros_like(logits_s).long()
    gt_mask[:, -1] = 1
    other_mask = 1 - gt_mask

    # gt_mask = _get_gt_mask(logits_s, labels)
    # other_mask = _get_other_mask(logits_s, labels)

    #norm
    tstd=logits_t.std(dim=1,keepdim=True)
    sstd=logits_s.std(dim=1,keepdim=True)
    dywt=tstd*t_norm
    dyws=sstd*t_norm
    rt=(logits_t)/dywt
    rs=(logits_s)/dyws

    pred_student = F.softmax(rs, dim=1)
    pred_teacher = F.softmax(rt, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student)

    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1,keepdim=True)*(dywt**2)
    
    pred_teacher_part2 = F.softmax(
        rt - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        rs - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none").sum(1,keepdim=True)*(dywt**2)
    
    if reduction == 'mean':
        tckd_loss = tckd_loss.mean()
        nckd_loss = nckd_loss.mean()

    return alpha*tckd_loss + beta*nckd_loss  
