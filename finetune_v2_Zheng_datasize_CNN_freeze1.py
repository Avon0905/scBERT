# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import logging
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch_v2 import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, metavar='N', help='Local process rank.')
args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = 2021
EPOCHS = int(100)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 32
LEARNING_RATE = 1e-4
SEQ_LEN = 16906 + 1
VALIDATE_EVERY = 1
SCHED = 'CAWUP'
UNASSIGN_THRES = 0.0

CLASS = 7
MASK_PROB = 0.15
REPLACE_PROB = 0.9
NUM_TOKENS = None
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]

log_step = 1000
model_name = f'Zheng_datasize5_CNN_freeze1'
ckpt_folder = '/aaa/jianhuayao/project2/performer-pytorch-main/examples/ddp_ckpts/'
log_folder = '/aaa/jianhuayao/project2/performer-pytorch-main/examples/ddp_logs/'
logger = set_log(logfileName=f'{log_folder}{model_name}', rank=local_rank)
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()
if is_master:
    logger.info(f'            =======  GPU num is {torch.distributed.get_world_size()}  ========= \n')
seed_all(SEED + torch.distributed.get_rank())
if is_master:
    logger.info('            =======  Config over  ======= \n')

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad('/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_log2_10k_aligned.h5ad')
# assigned = (np.where(np.array(data.obs['CellType'])!='Unassigned'))[0]
# data = data[assigned]
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)          # 将strings categorical 转化为 integrate categorical；label_dict[label]可以还原标签
class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X
if is_master:
    logger.info(f'    ==  Dim of dataset: {data.shape[0]}, {data.shape[1]}  == \n')

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
for s_i, s_j in sss.split(data,label):
    data_tmp, data_val, label_tmp, label_val = data[s_i], data[s_j], label[s_i], label[s_j]
# data_tmp, data_val, label_tmp, label_val = train_test_split(data, label, test_size=0.1,random_state=SEED)

accs = []
f1s = []
f1ws = []
for size_factor in [2e-4, 0.88, 0.66, 0.44, 0.22]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=size_factor, random_state=SEED)
    for s_i, s_j in sss.split(data_tmp, label_tmp):
        data_train, label_train = data_tmp[s_i], label_tmp[s_i]
    # data_train, _, label_train, _ = train_test_split(data_tmp, label_tmp, test_size=size_factor, random_state=SEED)
    if is_master:
        logger.info(f'            =======  Split over  ======= \n')

    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)
    if is_master:
        logger.info(f'            =======  Dataset over  ======= \n')

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    if is_master:
        logger.info(f'            =======  Sampler over  ======= \n')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    if is_master:
        logger.info(f'            =======  Dataloader over  ======= \n')


    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = False
    )
    if is_master:
        logger.info(f'            =======  Model defined  ======= \n')

    path = '/aaa/jianhuayao/project2/performer-pytorch-main/examples/ddp_ckpts/panglao_posFalse_CAWUP_7cls_5.pth'
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[-1].parameters():
        param.requires_grad = True
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if is_master:
        logger.info(f'            =======  Model distributed  ======= \n')
    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer2 = SGD(model.parameters(), lr=LEARNING_RATE)
    if SCHED == 'CYC':
        optimizer = optimizer2
    # learning rate scheduler
    schedules = {
        'STEP': StepLR(
            optimizer,
            step_size=1,
            gamma=0.9
        ),
        'CAW': CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        ),
        'CAWUP': CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=LEARNING_RATE,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        ),
        'CYC': CyclicLR(
            optimizer2,
            base_lr=1e-6,
            max_lr=LEARNING_RATE,
            step_size_up=5,
            mode="triangular2",
        )
    }
    scheduler = schedules[SCHED]
    loss_fn = nn.CrossEntropyLoss().to(local_rank)
    if is_master:
        logger.info(f'            =======  Optimizer is setup  ======= \n')

    dist.barrier()
    if is_master:
        logger.info('            =======  Training start  ======= \n')

    for i in range(1, EPOCHS+1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        logging_loss = 0.0
        cum_acc = 0.0
        logging_acc = 0.0
        for index, (data, labels) in enumerate(train_loader):
            index += 1
            data, labels = data.to(device), labels.to(device)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            if index % log_step == 0:
                logging_avg_loss = (running_loss - logging_loss) / log_step
                logging_loss = running_loss
                logging_avg_acc = 100 * (cum_acc - logging_acc) / log_step
                logging_acc = cum_acc
                logging_avg_loss = get_reduced(logging_avg_loss, local_rank, 0, world_size)
                logging_avg_acc = get_reduced(logging_avg_acc, local_rank, 0, world_size)
                if is_master:
                    print(f'    ==  Epoch: [{i}/{EPOCHS}] | \
                        Step: [{index}/{len(train_loader)}] | \
                        Training Loss: {logging_avg_loss:.6f} | \
                        Accuracy: {logging_avg_acc:6.4f}%  ==')
        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        if is_master:
            progress_train = {"step": i, "type": "train", "loss": epoch_loss, "acc": epoch_acc}
            report_progress(progress_train)
            logger.info(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()
            running_loss = 0.0
            logging_loss = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, (data, labels) in enumerate(val_loader):
                    index += 1
                    data, labels = data.to(device), labels.to(device)
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final_prob = softmax(logits)
                    final = final_prob.argmax(dim=-1)
                    final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                    # final[torch.amax(final_prob, axis=-1) < UNASSIGN_THRES] = -1 # 1.8.0才有torch.amax
                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final_prob, final
                # gather
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
                acc = accuracy_score(truths, predictions)
                f1 = f1_score(truths, predictions, average='macro')
                f1w = f1_score(truths, predictions, average='weighted', zero_division=0)
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                if is_master:
                    progress_val = {"step": i, "type": "test", "loss": val_loss, "f1_score": f1}
                    report_progress(progress_val)
                    logger.info(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
                    print(f'Unassigned Rate: {(len(val_sampler.dataset) - len(no_drop)) / len(val_sampler.dataset) * 100}%')
                    print(confusion_matrix(truths, predictions))
                    print((1-size_factor)*0.9)
                    print(classification_report(truths, predictions, target_names=label_dict.tolist(), digits=4))

        del predictions, truths
    accs.append(acc)
    f1s.append(f1)
    f1ws.append(f1w)
if is_master:
    logger.info(f'            =======  Training end  ======= \n')
if is_master:
    datasizes = ['0.9', '0.1', '0.3', '0.5', '0.7']
    print(datasize)
    print(accs)
    print(f1s)
    print(f1ws)
    with open('/aaa/jianhuayao/project2/other_methods/Zheng_datasize/finetune_datasize5.out', 'a') as fd:
        fd.write(','.join(datasizes)+'\n')
        fd.write(','.join(map(lambda x: str(x), accs))+'\n')
        fd.write(','.join(map(lambda x: str(x), f1s))+'\n')
        fd.write(','.join(map(lambda x: str(x), f1ws))+'\n')
if is_master:
    job_completed()

# for v in range(f3.shape[0]):
#     if v % 1000 == 0:
#         print(v)
#     for u in range(data.obs.shape[0]):
#         if f3.index[v] == data.obs.index[u]:
#             data.obs.loc[f3.index[v],'label'] = f3.loc[f3.index[v],'label']