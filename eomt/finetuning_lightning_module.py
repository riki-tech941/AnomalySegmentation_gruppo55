import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Literal, Optional
from training.lightning_module import LightningModule
from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule
from training.mask_classification_semantic import MaskClassificationSemantic


# Counts and prints the number of trainable and total parameters in the model

def _log_parameters(model) :
    trainable_parameters = 0
    total_parameters = 0
    for p in model.parameters():
        total_parameters += p.numel()
        if p.requires_grad == True:
            trainable_parameters += p.numel()
    percentuale = (trainable_parameters / total_parameters) * 100
    message = f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,} ({percentuale:.1f}%)"
    logging.info(message)



# Freeze the whole model except the head

HEAD_MODULES = ("class_head", "mask_head", "upscale", "q.weight")

def unfreeze_head(model):
    for name, parameter in model.named_parameters() :
        is_head_module = False
        for head_module in HEAD_MODULES :
            if head_module in name :
                is_head_module = True
                break
        if is_head_module :
            parameter.requires_grad = True
        else :
            parameter.requires_grad = False
    _log_parameters(model)



# Freeze the whole model except the last N blocks

def unfreeze_last_n_blocks(model, n):
    unfreeze_head(model)
    block_list = model.network.encoder.backbone.blocks
    starting_index = len(block_list) - n
    for name, parameter in model.named_parameters() :
        if "backbone.blocks" in name :
            block_index_string = name.split("backbone.blocks.")[1].split(".")[0]
            block_index = int(block_index_string)
            if block_index >= starting_index :
                parameter.requires_grad = True
            if "backbone.norm" in name :
                parameter.requires_grad = True
    _log_parameters(model)



# LightningModule for fine-tuning

    
class FinetuneLightningModule(MaskClassificationSemantic): 

        def __init__(
            self,
            mode: Literal["head", "last_n_blocks"] = "head",
            n_blocchi: int = 4,
            lr_head: float = 1e-4,
            lr_backbone: float = 1e-5,
            checkpoint_path: Optional[str] = None,
            load_ckpt_class_head: bool = True,
            **kwargs,
        ):
            super().__init__(
                load_ckpt_class_head = load_ckpt_class_head, 
                **kwargs)
            
            self.lr_head = lr_head
            self.lr_backbone = lr_backbone
            print("pos_embed shape:", self.network.encoder.backbone.pos_embed.shape)

            if mode == "head":
                unfreeze_head(self)
            elif mode == "last_n_blocks":
                unfreeze_last_n_blocks(self, n_blocchi)
            else:
                raise ValueError(f"Mode {mode} not valid")
            

        # Confugres the optimizer (AdamW) and the scheduler

        def configure_optimizers(self) :

            head_parameters = []
            backbone_parameters = []

            # Creates configuration dictionaries for the optimizer

            for name, parameter in self.named_parameters() :
                if parameter.requires_grad == False :
                    continue
                is_head_module = False
                for head_module in HEAD_MODULES :
                    if head_module in name :
                        is_head_module = True
                        break
                parameter_configuration = {
                    'name' : name,
                    'params' : [parameter],
                }
                if is_head_module :
                    parameter_configuration["lr"] = self.lr_head
                    head_parameters.append(parameter_configuration)
                else :
                    parameter_configuration["lr"] = self.lr_backbone
                    backbone_parameters.append(parameter_configuration)

            all_parameters = head_parameters + backbone_parameters

            optimizer = AdamW(
                all_parameters,
                weight_decay=self.weight_decay
            )

            scheduler = TwoStageWarmupPolySchedule(
                optimizer,
                num_backbone_params=len(backbone_parameters),
                warmup_steps=self.warmup_steps,
                total_steps=self.trainer.estimated_stepping_batches,
                poly_power=self.poly_power,
            )

            final_configuration = {
                'optimizer' : optimizer,
                'lr_scheduler' : {
                    'scheduler' : scheduler,
                    'interval' : 'step'
                }
            }
            return final_configuration



