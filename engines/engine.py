from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from datasets.multiple_datasets import MultipleDatasets, datasets_dict
from datasets.common import COMMON
from transformers import get_scheduler
from safetensors.torch import load_file
import os
import re
import time
import datetime
from models import build_dgg_model
from .funcs.eval_funcs import *
from utils.visualization import render_mesh
from utils.box_ops import box_cxcywh_to_xyxy
import cv2
from .funcs.infer_funcs import inference
from utils import misc
from utils.misc import get_world_size
import torch.multiprocessing
import numpy as np


class Engine():
    def __init__(self, args, mode='train'): 
        self.exp_name = args.exp_name
        self.mode = mode
        assert mode in ['train','eval','infer']
        self.conf_thresh = args.conf_thresh
        # Stage-1 depth-only mode flag (may not exist in all configs)
        self.depth_stage1_only = getattr(args, 'depth_stage1_only', False)

        # Default evaluation function mapping
        self.eval_func_maps = {'agora_validation': evaluate_agora,
                                'bedlam_validation_6fps': evaluate_agora,
                                'agora_test': test_agora,
                                '3dpw_test': evaluate_3dpw,
                                'cmu_test': evaluate_cmu,
                                'mupots_test': evaluate_mupots}

        # In depth_stage1_only mode, use a lightweight visualization-only
        # evaluation that plots GT/Pred bboxes and Tz without computing metrics.
        if self.depth_stage1_only:
            if 'agora_validation' in self.eval_func_maps:
                self.eval_func_maps['agora_validation'] = visualize_depth_stage1
            if 'bedlam_validation_6fps' in self.eval_func_maps:
                self.eval_func_maps['bedlam_validation_6fps'] = visualize_depth_stage1
        self.inference_func = inference

        if self.mode == 'train':
            self.output_dir = os.path.join('./outputs')
            self.log_dir = os.path.join(self.output_dir,'logs')
            self.ckpt_dir = os.path.join(self.output_dir,'ckpts')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'eval':
            self.output_dir = os.path.join('./results')
            self.distributed_eval = args.distributed_eval
            self.eval_vis_num = args.eval_vis_num
        elif self.mode == 'infer':
            output_dir = getattr(args, 'output_dir', None)
            if output_dir is not None:
                self.output_dir = output_dir
            else:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join('./results',f'{self.exp_name}_infer_{timestamp}')
            self.distributed_infer = args.distributed_infer

        self.prepare_accelerator()
        self.prepare_models(args)
        self.prepare_datas(args)
        if self.mode == 'train':
            self.prepare_training(args)

        total_cnt = sum(p.numel() for p in self.model.parameters())
        trainable_cnt = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.accelerator.print(f'Initialization finished.\n{trainable_cnt} trainable parameters({total_cnt} total).')       

    def prepare_accelerator(self):
        if self.mode == 'train':
            self.accelerator = Accelerator(
                log_with="tensorboard",
                project_dir=os.path.join(self.log_dir),
            )
            if self.accelerator.is_main_process:
                os.makedirs(self.log_dir, exist_ok=True)
                os.makedirs(os.path.join(self.ckpt_dir,self.exp_name),exist_ok=True)
                self.accelerator.init_trackers(self.exp_name)
        else:
            self.accelerator = Accelerator()
            if self.accelerator.is_main_process:
                os.makedirs(self.output_dir, exist_ok=True)

        if getattr(self, 'distributed_eval', False) and self.accelerator.num_processes <= 1:
            self.accelerator.print(
                'distributed_eval=True but only 1 process is running. '
                'To enable multi-GPU distributed evaluation, launch with: accelerate launch main.py --mode eval --cfg <cfg_name>'
            )
        
    def prepare_models(self, args):
        # load model and criterion
        self.accelerator.print('Preparing models...')
        self.unwrapped_model, self.criterion = build_dgg_model(args, set_criterion = (self.mode == 'train'))
        if self.criterion is not None:
            self.full_weight_dict = self.criterion.weight_dict
            self.weight_dict = self.full_weight_dict.copy()
        # load weights
        if args.pretrain:
            self.accelerator.print(f'Loading pretrained weights: {args.pretrain_path}') 
            state_dict = torch.load(args.pretrain_path, weights_only=False)
            self.unwrapped_model.load_state_dict(state_dict, strict=False)

        # to gpu
        self.model = self.accelerator.prepare(self.unwrapped_model)
        
    def prepare_datas(self, args):
        # load dataset and dataloader
        if self.mode == 'train':
            self.accelerator.print('Loading training datasets:\n',
                            [f'{d}_{s}' for d,s in zip(args.train_datasets_used, args.train_datasets_split)])
            self.train_batch_size = args.train_batch_size
            train_dataset = MultipleDatasets(args.train_datasets_used, args.train_datasets_split,
                                        make_same_len=False, input_size=args.input_size, aug=True, 
                                        mode = 'train', aug_cfg=args.aug_cfg)
            self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                                shuffle=True,collate_fn=misc.collate_fn, 
                                                num_workers=args.train_num_workers,pin_memory=True)
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)                                 

        if self.mode != 'infer':
            self.accelerator.print('Loading evaluation datasets:',
                                [f'{d}_{s}' for d,s in zip(args.eval_datasets_used, args.eval_datasets_split)])
            self.eval_batch_size = args.eval_batch_size
            eval_ds = {f'{ds}_{split}': datasets_dict[ds](split = split, 
                                                          mode = 'eval', 
                                                          input_size = args.input_size, 
                                                          aug = False)\
                        for (ds, split) in zip(args.eval_datasets_used, args.eval_datasets_split)}
            self.eval_dataloaders = {k: DataLoader(dataset=v, batch_size=self.eval_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.eval_num_workers,pin_memory=True)\
                                    for (k,v) in eval_ds.items()}
            if self.distributed_eval:
                for k in list(self.eval_dataloaders.keys()):
                    self.eval_dataloaders[k] = self.accelerator.prepare(self.eval_dataloaders[k])
        
        else:
            img_folder = args.input_dir
            self.accelerator.print(f'Loading inference images from {img_folder}')
            self.infer_batch_size = args.infer_batch_size
            infer_ds = COMMON(img_folder = img_folder, input_size=args.input_size,aug=False,
                                mode = 'infer')
            self.infer_dataloader = DataLoader(dataset=infer_ds, batch_size=self.infer_batch_size,
                                        shuffle=False,collate_fn=misc.collate_fn, 
                                        num_workers=args.infer_num_workers,pin_memory=True)

            if self.distributed_infer:
                self.infer_dataloader = self.accelerator.prepare(self.infer_dataloader)

    def prepare_training(self, args):
        self.start_epoch = 0
        self.num_epochs = args.num_epochs
        self.global_step = 0
        # No extra epoch-specific GT switching is used.
        self.save_and_eval_epoch = args.save_and_eval_epoch
        self.least_eval_epoch = args.least_eval_epoch

        self.detach_j3ds = args.detach_j3ds

        self.accelerator.print('Preparing optimizer and lr_scheduler...')   
        param_dicts = [
            {
                "params":
                    [p for n, p in self.unwrapped_model.named_parameters()
                    if not misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": 
                    [p for n, p in self.unwrapped_model.named_parameters() 
                    if misc.match_name_keywords(n, args.lr_encoder_names) and p.requires_grad],
                "lr": args.lr_encoder,
            }
        ]

        # optimizer
        if args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        
        # lr_scheduler
        if args.lr_scheduler == 'cosine':
            self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer, 
                                          num_warmup_steps=args.num_warmup_steps, 
                                          num_training_steps=get_world_size() * self.num_epochs * len(self.train_dataloader)) 
        elif args.lr_scheduler == 'multistep':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, gamma=args.gamma)  
        else:
            raise NotImplementedError      

        self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # resume
        if args.resume: #load model, optimizer, lr_scheduler and random_state
            if hasattr(args, 'ckpt_epoch'):
                self.load_ckpt(args.ckpt_epoch,args.ckpt_step)    
            else:
                self.accelerator.print('Auto resume from the latest ckpt...')
                epoch, step = -1, -1
                pattern = re.compile(r'epoch_(\d+)_step_(\d+)')
                for folder_name in os.listdir(os.path.join(self.output_dir,'ckpts',self.exp_name)):
                    match = pattern.match(folder_name)
                    if match:
                        i, j = int(match.group(1)), int(match.group(2))
                        if i > epoch:
                            epoch, step = i, j
                if epoch >= 0:
                    self.load_ckpt(epoch, step)    
                else:
                    self.accelerator.print('No existing ckpts! Train from scratch.')               

    def load_ckpt(self, epoch, step):   
        self.accelerator.print(f'Loading checkpoint: epoch_{epoch}_step_{step}') 
        ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{step}')
        self.start_epoch = epoch + 1
        self.global_step = step + 1
        self.accelerator.load_state(ckpts_save_path)

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.accelerator.print('Start training!')
        for epoch in range(self.start_epoch, self.num_epochs):
            torch.cuda.empty_cache()
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            current_weight_dict = self.full_weight_dict

            self.model.train()
            self.criterion.train()

            for step, (samples,targets) in enumerate(self.train_dataloader):
                outputs = self.model(samples, targets, detach_j3ds = self.detach_j3ds)
                loss_dict = self.criterion(outputs, targets, step=self.global_step)

                loss = sum(loss_dict[k] * current_weight_dict[k] for k in loss_dict.keys())

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                reduced_dict = self.accelerator.reduce(loss_dict,reduction='mean')
                simplified_logs = {k: v.item() for k, v in reduced_dict.items() if '.' not in k}
                
                # logs.update({"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})
                if self.accelerator.is_main_process:
                    tqdm.write(f'[{epoch}-{step+1}/{len(self.train_dataloader)}]: ' + str(simplified_logs))

                if step % 10 == 0:
                    log_dict = {('train/'+k):v for k,v in simplified_logs.items()}
                    self.accelerator.log(log_dict, step=self.global_step)
                
                progress_bar.update(1)
                progress_bar.set_postfix(**{"lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step})

                self.global_step += 1

                self.accelerator.wait_for_everyone()

            # self.lr_scheduler.step()

            if epoch % self.save_and_eval_epoch == 0 or epoch == self.num_epochs-1:
                self.save_and_eval(epoch, save_ckpt=True)
        
        self.accelerator.end_training()

    def eval(self, results_save_path = None, epoch = -1):
        if results_save_path is None:
            results_save_path = os.path.join(self.output_dir,self.exp_name,'evaluation')
        # preparing
        self.model.eval()
        unwrapped_model = self.unwrapped_model # self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        # evaluate
        for i, (key, eval_dataloader) in enumerate(self.eval_dataloaders.items()):
            assert key in self.eval_func_maps
            img_cnt = len(eval_dataloader) * self.eval_batch_size
            if self.distributed_eval:
                img_cnt *= self.accelerator.num_processes
            self.accelerator.print(f'Evaluate on {key}: {img_cnt} images')
            self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
            conf_thresh = self.conf_thresh  # if 'agora' in key or 'bedlam' in key else [0.2]
            for thresh in conf_thresh:
                if self.accelerator.is_main_process or self.distributed_eval:
                    error_dict = self.eval_func_maps[key](model = unwrapped_model, 
                                    eval_dataloader = eval_dataloader, 
                                    conf_thresh = thresh,
                                    vis_step = img_cnt // self.eval_vis_num,
                                    results_save_path = os.path.join(results_save_path,key,f'thresh_{thresh}'),
                                    distributed = self.distributed_eval,
                                    accelerator = self.accelerator,
                                    vis=True)
                    if isinstance(error_dict,dict) and self.mode == 'train':
                        log_dict = flatten_dict(error_dict)
                        self.accelerator.log({(f'{key}_thresh_{thresh}/'+k):v for k,v in log_dict.items()}, step=epoch)

                    self.accelerator.print(f'thresh_{thresh}: ',error_dict)
                self.accelerator.wait_for_everyone() 
  
    def save_and_eval(self, epoch, save_ckpt=False):
        torch.cuda.empty_cache()
        # save current state and model
        if self.accelerator.is_main_process and save_ckpt:
            ckpts_save_path = os.path.join(self.output_dir,'ckpts',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')
            os.makedirs(ckpts_save_path,exist_ok=True)
            self.accelerator.save_state(ckpts_save_path, safe_serialization=False)
        self.accelerator.wait_for_everyone()
        
        if epoch < self.least_eval_epoch:
            return
        results_save_path = os.path.join(self.output_dir,'results',self.exp_name, f'epoch_{epoch}_step_{self.global_step-1}')        
        self.eval(results_save_path, epoch=epoch)

    def infer(self):
        self.model.eval()
        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = self.unwrapped_model 
        
        results_save_path = self.output_dir
        if self.accelerator.is_main_process:
            os.makedirs(results_save_path,exist_ok=True)
        
        self.accelerator.print('Using following threshold(s): ', self.conf_thresh)
        for thresh in self.conf_thresh:
            if self.accelerator.is_main_process or self.distributed_infer:
                self.inference_func(model = unwrapped_model, 
                        infer_dataloader = self.infer_dataloader, 
                        conf_thresh = thresh,
                        results_save_path = os.path.join(results_save_path,f'thresh_{thresh}'),
                        distributed = self.distributed_infer,
                        accelerator = self.accelerator)
            self.accelerator.wait_for_everyone()


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)






