from config import *
from collections import Counter
from typing import Tuple, Optional, Union
import math
from distil_losses import *


def mse_loss(logits_s, logits_t, temperature=1):
    num_pos_to_compute_loss = 1024
    beta_logits_t = logits_t
    beta_logits_s = logits_s

    with torch.no_grad():
        mask_pos = torch.argsort(beta_logits_t, dim=-1)
        select_pos = mask_pos[:,:, -num_pos_to_compute_loss:]
        select_logits_t = torch.gather(beta_logits_t, 2, select_pos)
        # select_logits_t = re_arange(select_logits_t)

    select_logits_s = torch.gather(beta_logits_s, 2, select_pos)
    loss = \
    F.mse_loss(select_logits_s.softmax(-1), select_logits_t.softmax(-1), reduction='none')
    return loss


def normed_ce_loss(logits, labels):
    
    batch_size, seq_length, vocab_size = logits.shape
    
    # return F.cross_entropy(logits.view(-1, vocab_size), 
    #                     labels.view(-1))
    
    loss_mask = (labels > -1).long().float().to(logits.device)
    expected_number_of_tokens = loss_mask.sum()
    loss_mask /= (loss_mask.sum(1).unsqueeze(1) + 1e-8)
    loss_mask = loss_mask.view(-1)

    # expected_number_of_tokens = ((1+seq_length)/2) * batch_size
    loss_func = nn.CrossEntropyLoss(reduction='none')
    lm_loss = loss_func(logits.view(-1, vocab_size), 
                        labels.view(-1)
                        )
    lm_loss = (lm_loss.view(-1) * loss_mask).sum() / batch_size
    return lm_loss

def topk_normed_ce_loss(logits, labels, weights=None):
    
    num_pos_to_compute_loss = 1024

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
        select_pos = mask_pos[:,:,-num_pos_to_compute_loss:]
    select_logits = torch.gather(logits, 2, select_pos)

    loss_mask = (labels > -1).long().float().to(logits.device)
    loss_mask /= (loss_mask.sum(1, keepdim=True) + 1e-8)
    if weights is not None:
        loss_mask *= weights.unsqueeze(1)
    loss_mask = loss_mask.view(-1)
    fake_labels = torch.ones_like(loss_mask) * (num_pos_to_compute_loss-1)
    fake_labels = fake_labels.long()

    loss_func = nn.CrossEntropyLoss(reduction='none')

    lm_loss = loss_func(select_logits.view(-1, num_pos_to_compute_loss), 
                        fake_labels.view(-1)
                        )
    lm_loss = (lm_loss.view(-1) * loss_mask).sum() / batch_size
    return lm_loss


def re_arange(logits, ratio = 1):
    '''
    could 1/ratio works the same with temperature?
    '''
    with torch.no_grad():
        shape = logits.shape
        logits = logits.reshape(-1, logits.shape[-1])
        sorted_idxs = torch.argsort(logits, dim=-1)
        target_idxs_for_probs = torch.argsort(sorted_idxs, dim=-1)
        min_vals = logits.min(-1, keepdim=True)[0]
        logits_from_zero = logits - min_vals
        max_vals = logits_from_zero.max(-1, keepdim=True)[0]
        def func(x, a=1, b=1, c=0):
            return a * ((b*x)**4) + c
        x = []
        x_ends = torch.pow(max_vals, 1/4)
        for idx in range(x_ends.shape[0]):
            x.append(torch.linspace(0, x_ends[idx].item(), shape[-1]))
        x = torch.stack(x, 0)
        candicate_new_logits = func(x)
        candicate_new_logits = candicate_new_logits.to(logits.device)
        
        new_logits = torch.gather(candicate_new_logits, 1, target_idxs_for_probs)
        new_logits *= ratio
        new_logits += min_vals
        new_logits = new_logits.reshape(*shape)
        return new_logits


def correct_logits(logits, labels):
    '''
    logits: batch, seq, vocab_size
    labels: batch, seq
    '''
    batch_size, max_seq_len, vocab_size = logits.shape
    reshape_logits = logits.reshape(-1, vocab_size)
    reshape_labels = labels.reshape(-1)
    for idx in range(batch_size * max_seq_len):
        label = reshape_labels[idx]
        if label < 0:
            continue
        pred = torch.argmax(reshape_logits[idx])
        if pred != label:
            # TODO check correct
            t = reshape_logits[idx][label]
            # more sharp
            reshape_logits[idx][label] = reshape_logits[idx].max().item()* 2
            reshape_logits[idx][pred] = t
    return reshape_logits.reshape(batch_size, max_seq_len, vocab_size)

def mix(logits_s, logits_t, alpha=0.8):
    with torch.no_grad():
        batch_size, max_seq_len, vocab_size = logits_s.shape
        t_min = logits_t.min(-1)[0].unsqueeze(-1).expand_as(logits_t)
        t_max = logits_t.max(-1)[0].unsqueeze(-1).expand_as(logits_t)
        s_min = logits_s.min(-1)[0].unsqueeze(-1).expand_as(logits_s)
        s_max = logits_s.max(-1)[0].unsqueeze(-1).expand_as(logits_s)
        scaled_logits_s = (logits_s - s_min ) / (s_max - s_min) # 0..1
        scaled_logits_s *= (t_max - t_min)
        scaled_logits_s += t_min
        mix_logits_t = logits_t * alpha + (1-alpha) * scaled_logits_s
    return mix_logits_t

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = transformers.trainer_pt_utils.get_parameter_names(
            model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    def _create_optimizer_(self):# disabled
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if transformers.utils.is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'lm_head' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'lm_head' not in n)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            for n, p in opt_model.named_parameters():
                if 'lm_head' in n:
                    optimizer_grouped_parameters.append(
                        {
                            'params': p,
                            'lr': self.args.learning_rate,
                            'weight_decay': self.args.weight_decay if n in decay_parameters else 0.0
                        }
                    )
                    break

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


class CustomTrainerForSFT(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine_with_restarts':
                self.lr_scheduler = transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    num_cycles=self.args.max_steps // self.args.save_steps,
                    last_epoch=-1
                )
            else:
                self.lr_scheduler = transformers.optimization.get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        weights = inputs.get('weights')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = topk_normed_ce_loss(shift_logits, shift_labels, weights=weights)
        # lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), 
        #                           shift_labels.view(-1), reduction='mean', weight=weights)
        # print0(lm_loss.item())
        return (lm_loss, outputs) if return_outputs else lm_loss

class CustomTrainerForDistillation(CustomTrainerForSFT):
    def __init__(self, config, teacher_model_path, temperature=3.0, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.loss_type = self.config['loss_type']
        self.temperature = temperature
        self.teacher = None
        self.teacher_model_path = teacher_model_path

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get('input_ids')
        labels = inputs.get("labels")
        attention_mask = inputs.get('attention_mask')

        shift_labels = labels[..., 1:].contiguous()
        weights = inputs.get('weights')

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits_s = student_outputs.logits.contiguous()

        if self.teacher is None:
            self.teacher = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_path, trust_remote_code=True, low_cpu_mem_usage=True
                )
            self.teacher.eval()
        if self.teacher.device != model.device:
            self.teacher.to(model.device)

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        logits_t =  teacher_outputs['logits'].detach().contiguous()

        logits_s = logits_s[..., :-1, :].contiguous()
        logits_t = logits_t[..., :-1, :].contiguous()
        attention_mask = attention_mask[..., :-1].contiguous()


        # instruction loss 
        instruction_temprature = 1
        distil_loss_mask = attention_mask * (shift_labels < 0).long().float()
        # distil_loss_mask /= (distil_loss_mask.sum(1, keepdim=True) + 1e-8)
        distil_loss_mask = distil_loss_mask.view(-1)
        distil_loss1 = vanilla_kl_loss_func(logits_s, logits_t, temperature=instruction_temprature)
        instruction_loss = (distil_loss1.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        # distil loss mask
        distil_loss_mask = (shift_labels > -1).long()
        distil_loss_mask = distil_loss_mask.long().float()
        # distil_loss_mask /= (distil_loss_mask.sum(1, keepdim=True) + 1e-8)
        distil_loss_mask = distil_loss_mask.view(-1)

        if self.loss_type == 'vanilla_kl':
            vanilla_kl_loss = vanilla_kl_loss_func(logits_s, logits_t, temperature=self.temperature)
            vanilla_kl_loss = (vanilla_kl_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        if self.loss_type == 'rkl':
            reverse_kl_loss = reverse_kl_loss_func(logits_s, logits_t, temperature=self.temperature)
            reverse_kl_loss = (reverse_kl_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        if self.loss_type == 'top_kl':
            top_kl_loss = top_kl_loss_func(logits_s, logits_t, self.temperature, top_k=self.config['top_kl_k'])
            top_kl_loss = (top_kl_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        if self.loss_type == 'bild':
            t_ld_loss = bild_loss_func(logits_s, logits_t, top_k=self.config['bild_topk'], temperature=self.temperature, student_led=False)
            s_ld_loss = bild_loss_func(logits_s, logits_t, top_k=self.config['bild_topk'], temperature=self.temperature, student_led=True)
            t_ld_loss = (t_ld_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()
            s_ld_loss = (s_ld_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()
            bild_loss = t_ld_loss + s_ld_loss

        # DKD loss
        if self.loss_type == 'dkd':
            dkd_loss = dkd_loss_func(logits_s, logits_t, labels=shift_labels, t_norm=self.temperature)
            dkd_loss = (dkd_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        # NKD loss
        if self.loss_type == 'nkd':
            nkd_loss = nkd_loss_func(logits_s, logits_t, temperature=self.temperature)
            nkd_loss = (nkd_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

        # NormKD loss
        if self.loss_type == 'normkd':
            normkd_loss = normkd_loss_func(logits_s, logits_t, T_norm=self.temperature)
            normkd_loss = (normkd_loss.view(-1) * distil_loss_mask).sum() / distil_loss_mask.sum()

    
        distil_loss = 0
        distil_loss += instruction_loss

        if self.loss_type == 'vanilla_kl':
            distil_loss += vanilla_kl_loss
        elif self.loss_type == 'rkl':
            distil_loss += reverse_kl_loss
        elif self.loss_type == 'top_kl':
            distil_loss += top_kl_loss
        elif self.loss_type == 'bild':
            distil_loss += bild_loss
        elif self.loss_type == 'dkd':
            distil_loss += dkd_loss
        elif self.loss_type == 'nkd':
            distil_loss += nkd_loss
        elif self.loss_type == 'normkd':
            distil_loss += normkd_loss
        
        loss = distil_loss

        return (loss, student_outputs) if return_outputs else loss