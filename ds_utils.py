# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": False,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True,
    }
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "zero_optimization": zero_opt_dict,
        "bf16":{
            "enabled" : "auto"
        },
        "postscale_gradients": True,
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"
            }
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": 1e-8,
            "weight_decay": "auto"
            }
        },
    }
