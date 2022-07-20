# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# Copyright (c) 2022 Christopher Thomas, Yipeng Zhang, Shih-Fu Chang. See acknowledgement for terms of use.
# Modified from https://github.com/microsoft/Oscar

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import base64
import copy, time, json
import pickle, collections
from tqdm import tqdm
import json
import re

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from oscar.modeling.modeling_bert import VEBert
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule

from oscar.utils.misc import set_seed
from oscar.utils.data_utils import VEDataset
from oscar.utils.task_utils import output_modes, processors

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, VEBert, BertTokenizer),
}


log_json = []

# Calculates the KE (MIL) and structural losses
def calculate_losses(args, tokenizer, model, node_edge_indices, hidden_states, sep_idxs, batch_labels, pair_ids, img_len, tag2region=None):
    ke_loss = torch.zeros(1).cuda()
    struc_loss = torch.zeros(1).cuda()
    structural_losses = []

    if args.ke_loss or args.struc_loss:
        sample_cnt = 0

        for pid_i, pid in enumerate(pair_ids):
            img_id = pid.split('.')[0]

            if not node_edge_indices[pid]['node_indices']: 
                continue
                
            node_indices = node_edge_indices[pid]['node_indices']
            edge_indices = node_edge_indices[pid]['edge_indices']
            edges = node_edge_indices[pid]['edges']

            hyp_hidden_states = hidden_states[pid_i, 1:sep_idxs[3*pid_i, 1], :]
            if args.classifier == 'att+region':
                tag_hidden_states = hidden_states[pid_i, sep_idxs[3*pid_i, 1]+1:sep_idxs[3*pid_i+1, 1], :]
                end_img_idx = None if not -args.max_img_seq_length+img_len[pid_i].item() else -args.max_img_seq_length+img_len[pid_i].item()
                img_hidden_states = hidden_states[pid_i, -args.max_img_seq_length:end_img_idx, :]

            hyp_len = hyp_hidden_states.size(0)

            node_feats = []
            edge_feats = []

            ##### Remove indices that are outside of the sequence
            for i in range(len(node_indices)):
                node_indices[i] = [x for x in node_indices[i] if x < hyp_len]
            for i in range(len(edge_indices)):
                edge_indices[i] = [x for x in edge_indices[i] if x < hyp_len]
            for e in list(edges.keys()):
                if not edge_indices[e]:
                    del edges[e]

            ###### Contextual feature for node KEs
            valid_chunks = []
            node_removed = False
            for i, chunk in enumerate(node_indices):
                if not chunk:
                    node_removed = True
                    if args.classifier == 'att+region':
                        node_feats.append(torch.zeros((hyp_hidden_states.size(1)*2)).cuda())
                    else:
                        node_feats.append(torch.zeros_like(hyp_hidden_states[0, :]))
                    continue
                valid_chunks.append(i)

                chunk_feat = hyp_hidden_states[chunk, :]
                # linear projection from the token features to their attention weights
                alignment_score = model.module.ke_query(chunk_feat)
                node_feat = torch.sum(torch.softmax(alignment_score, dim=0)*chunk_feat, dim=0)

                if args.classifier == 'att+region':
                    if tag_hidden_states.size(0) == 0:
                        obj_feat = img_hidden_states[0]
                    else:
                        salient_node_feat = chunk_feat.detach()[torch.argmax(alignment_score.detach())]
                        # Compute similarity for all tags
                        sim = F.cosine_similarity(salient_node_feat, tag_hidden_states.detach(), dim=-1)
                        # Only compare tags that point to valid object regions (i.e., objects that are not truncated by the model)
                        valid_tag_idx = [t_i for t_i in range(sim.size(0)) if tag2region[img_id][t_i] < img_len[pid_i].item()]
                        tag_idx = valid_tag_idx[torch.argmax(sim[valid_tag_idx])]
                        # Retrieve object feature and concatenate
                        obj_idx = tag2region[img_id][tag_idx]
                        obj_feat = img_hidden_states[obj_idx]
                    node_feats.append(torch.cat([node_feat, obj_feat])) # contextualized node embedding

                elif args.classifier == 'att':
                    node_feats.append(node_feat) 

            if not valid_chunks:
                continue

            sample_cnt += 1  
            node_feats = torch.stack(node_feats, dim=0)

            ##### Contextual feature for tuple KEs
            relations = [] # node-edge-node tuple's token indices
            for e, n in edges.items():
                relations.append(node_indices[n[0]] + edge_indices[e] + node_indices[n[1]])     # node_indices[n[1]] can be empty
            
            for chunk in relations:
                chunk_feat = hyp_hidden_states[chunk, :]
                # linear projection from the token features to their attention weights
                alignment_score = model.module.ke_query(chunk_feat)
                edge_feat = torch.sum(torch.softmax(alignment_score, dim=0)*chunk_feat, dim=0)

                if args.classifier == 'att+region':
                    if tag_hidden_states.size(0) == 0: # no object tags (truncated or no detection), take the first object region
                        obj_feat = img_hidden_states[0]
                    else:
                        important_edge_feat = chunk_feat.detach()[torch.argmax(alignment_score.detach())]
                        # Compute similarity for all tags
                        sim = F.cosine_similarity(important_edge_feat, tag_hidden_states.detach(), dim=-1)
                        # Only compare tags that point to valid object regions (i.e., objects that are not truncated by the model)
                        valid_tag_idx = [t_i for t_i in range(sim.size(0)) if tag2region[img_id][t_i] < img_len[pid_i].item()]
                        tag_idx = valid_tag_idx[torch.argmax(sim[valid_tag_idx])]
                        # Retrieve object feature and concatenate
                        obj_idx = tag2region[img_id][tag_idx]
                        obj_feat = img_hidden_states[obj_idx]
                    edge_feats.append(torch.cat([edge_feat, obj_feat])) # contextualized tuple embedding

                elif args.classifier == 'att':
                    edge_feats.append(edge_feat)
            
            chunk_label = batch_labels[pid_i].item()  # Sample-level label as weak supervision

            # Obtain MLP Classifier logits
            if relations:
                edge_feats = torch.stack(edge_feats, dim=0)
                node_logits = model.module.ke_classifier(node_feats)
                edge_logits = model.module.ke_classifier(edge_feats)
                chunk_logits = torch.cat([node_logits[valid_chunks, :], edge_logits], dim=0)
            else:
                node_logits = model.module.ke_classifier(node_feats)
                chunk_logits = node_logits[valid_chunks, :]

            ############################### MIL Constraints ###############################

            # ENTAILMENT - ALL MUST BE PREDICTED ENTAILMENT
            if chunk_label == 0: 
                chunk_target = torch.Tensor([1, 0, 0]).float().cuda().expand_as(chunk_logits)
                ke_loss += F.binary_cross_entropy_with_logits(input=chunk_logits, target=chunk_target)
            
            # NEUTRAL - AT LEAST ONE PREDICTED NEUTRAL, NO CONTRADICTION
            elif chunk_label == 1: 
                # Enforce no contradiction
                chunk_target = torch.Tensor([0]).float().cuda().expand_as(chunk_logits[:, 2])
                ke_loss += F.binary_cross_entropy_with_logits(input=chunk_logits[:, 2], target=chunk_target)
                # Enforce at least one neutral
                if not node_removed:
                    max_idx = torch.argmax(chunk_logits[:, 1])
                    chunk_target = torch.Tensor([0, 1, 0]).float().cuda().expand_as(chunk_logits[max_idx, :])
                    ke_loss += F.binary_cross_entropy_with_logits(input=chunk_logits[max_idx,:], target=chunk_target)
            
            # CONTRADICTION - AT LEAST ONE PREDICTED CONTRADICTION
            elif chunk_label == 2: 
                # Enforce at least one contradiction
                if not node_removed:
                    max_idx = torch.argmax(chunk_logits[:, 2])
                    chunk_target = torch.Tensor([0, 0, 1]).float().cuda().expand_as(chunk_logits[max_idx, :])
                    ke_loss += F.binary_cross_entropy_with_logits(input=chunk_logits[max_idx,:], target=chunk_target)

            ############################### Structural Constraints ###############################
            if args.struc_loss and relations:
                node_preds = node_logits.argmax(dim=1)
                edge_preds = edge_logits.argmax(dim=1)
                node_pred_probs = torch.sigmoid(node_logits) # allow BP
                edge_pred_probs = torch.sigmoid(edge_logits) # allow BP

                for e, n in edges.items():

                    #### Top-Down Constraints ####
                    relation_pred = edge_preds[e].item()
                    relation_pred_prob = edge_pred_probs[e, relation_pred]
                    if relation_pred == 0: 
                        # PARENT IS ENTAILMENT, MUST BE E
                        structural_losses.append(relation_pred_prob*F.binary_cross_entropy_with_logits(
                            input=node_logits[n[0], :], target=torch.Tensor([1, 0, 0]).float().cuda()))
                        if node_indices[n[1]]:
                            structural_losses.append(relation_pred_prob*F.binary_cross_entropy_with_logits(
                                input=node_logits[n[1], :], target=torch.Tensor([1, 0, 0]).float().cuda()))
                    elif relation_pred == 1: 
                        # PARENT IS NEUTRAL, MUST NOT BE C
                        structural_losses.append(relation_pred_prob*F.binary_cross_entropy_with_logits(
                            input=node_logits[[n[0]], 2], target=torch.Tensor([0]).float().cuda()))
                        if node_indices[n[1]]:
                            structural_losses.append(relation_pred_prob*F.binary_cross_entropy_with_logits(
                                input=node_logits[[n[1]], 2], target=torch.Tensor([0]).float().cuda()))
                    elif relation_pred == 2: 
                        # PARENT IS CONTRADICTION, NO CONSTRAINT
                        pass

                    #### Bottom-up Constraints ####
                    for n_i in [0, 1]:
                        if not node_indices[n[n_i]]:
                            continue
                        n_pred = node_preds[n[n_i]].item()
                        n_pred_prob = node_pred_probs[n[n_i], n_pred]   # allow BP
                        if n_pred == 0: 
                            # CHILD IS ENTAILMENT, NO CONSTRAINT
                            pass
                        elif n_pred == 1: 
                            # CHILD IS NEUTRAL, PARENT NOT ENTAILMENT
                            structural_losses.append(n_pred_prob*F.binary_cross_entropy_with_logits(
                                    input=edge_logits[[e], 0], target=torch.Tensor([0]).float().cuda()))
                        elif n_pred == 2: 
                            # CHILD IS CONTRADICTION, PARENT MUST BE C
                            structural_losses.append(n_pred_prob*F.binary_cross_entropy_with_logits(
                                    input=edge_logits[e, :], target=torch.Tensor([0, 0, 1]).float().cuda())) 

        ke_loss = ke_loss / sample_cnt  # KE loss is averaged across samples

    if args.struc_loss:
        # structural loss is averaged across relations
        struc_loss = torch.mean(torch.stack(structural_losses)) if structural_losses else torch.zeros(1).cuda()

    return ke_loss, struc_loss


def train(args, train_dataset, eval_dataset, model, tokenizer, node_edge_indices, tag2region=None):

    manual_loss_weights = torch.Tensor(args.loss_weights).cuda()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=args.workers, sampler=train_sampler, batch_size=args.train_batch_size) #, collate_fn=trim_batch)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.lr_decay:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training 
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args.seed, args.n_gpu)  # Added here for reproductibility

    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    loss_history = {'cls': [], 'ke': [], 'struc': [], 'acc': []}

    t = 0
    for epoch in range(int(args.num_train_epochs)):

        for g_i, group in enumerate(optimizer.param_groups):
            print('Param Group {}: size={}, lr={}, wd={}'.format(g_i, len(group['params']), group['lr'], group['weight_decay']))

        total_loss = 0

        total_cls_loss = 0
        total_struc_loss = 0
        total_ke_loss = 0

        t_start = time.time()
        for step, batch in enumerate(train_dataloader):
    
            t += 1
            model.train()

            pair_ids = batch[1]
            img_len = batch[2]
            batch = tuple(t.to(args.device) for t in batch[0])
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3],
                      'img_feats':      batch[4]
            }
            outputs = model(**inputs)

            cls_loss, logits = outputs[:2]
            if args.n_gpu > 1: cls_loss = cls_loss.mean() # mean() to average on multi-gpu parallel training

            # shape (3*batch_size, 2)
            sep_idxs = torch.nonzero(inputs['input_ids'] == tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0])

            ke_loss, struc_loss = calculate_losses(args, tokenizer, model, node_edge_indices, \
                hidden_states=outputs[2], sep_idxs=sep_idxs, batch_labels=inputs['labels'], pair_ids=pair_ids, img_len=img_len, tag2region=tag2region)
            losses = torch.cat([cls_loss.view(1), ke_loss.view(1), struc_loss.view(1)])
            loss = (manual_loss_weights * losses).sum()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            total_loss += loss.item()

            total_cls_loss += cls_loss.item()
            total_ke_loss += ke_loss.item()
            total_struc_loss += struc_loss.item()

            loss_history['cls'].append(float(cls_loss.item()))
            loss_history['ke'].append(float(ke_loss.item()))
            loss_history['struc'].append(float(struc_loss.item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if args.lr_decay:
                    scheduler.step() 
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:# Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        logger.info("Epoch: %d, global_step: %d" % (epoch, global_step))
                        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
                        if eval_score > best_score:
                            best_score = eval_score
                            best_model['epoch'] = epoch
                            best_model['model'] = copy.deepcopy(model)

                        loss_history['acc'].append(float(eval_score))

                        logger.info("EVALERR: {}%".format(100 * best_score))
                    logging_loss = tr_loss

        t_end = time.time()
        logger.info('Train Time Cost: %.3f' % (t_end-t_start))

        # evaluation
        logger.info("Epoch: %d" % (epoch))
        eval_result, eval_score = evaluate(args, model, eval_dataset, prefix=global_step)
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch
            best_model['model'] = copy.deepcopy(model)
            #best_model['optimizer'] = copy.deepcopy(optimizer.state_dict())

        # save checkpoints
        if args.local_rank in [-1, 0] and args.save_epoch > 0 and epoch % args.save_epoch == 0: # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

            save_num = 0
            while (save_num < 10):
                try:
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    break
                except:
                    save_num += 1
            logger.info("Saving model checkpoint {0} to {1}".format(epoch, output_dir))

        epoch_log = {'epoch': epoch, 
            'total_loss': total_loss/len(train_dataloader), 
            'cls_loss': total_cls_loss/len(train_dataloader),
            'ke_loss': total_ke_loss/len(train_dataloader),
            'struc_loss': total_struc_loss/len(train_dataloader),
            'eval_score': eval_score, 'best_score':best_score}
        log_json.append(epoch_log)

        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
            json.dump(log_json, fp)

        with open(args.output_dir + '/loss_logs.pkl', 'wb') as fp:
            pickle.dump(loss_history, fp)

        logger.info("PROGRESS: {}%".format(round(100*(epoch + 1) / args.num_train_epochs, 4)))
        logger.info("EVALERR: {}%".format(100*best_score))
        logger.info("LOSS: {}%".format(total_loss / len(train_dataloader)))
        logger.info("CLS LOSS: {}%".format(total_cls_loss / len(train_dataloader)))
        logger.info("KE LOSS: {}%".format(total_ke_loss / len(train_dataloader)))
        logger.info("STRUCTURE LOSS: {}%".format(total_struc_loss / len(train_dataloader)))

    with open(args.output_dir + '/eval_logs.json', 'w') as fp:
        json.dump(log_json, fp)

    if args.local_rank in [-1, 0]: # Save the final model checkpoint
        output_dir = os.path.join(args.output_dir, 'best-{}'.format(best_model['epoch']))
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(model, 'module') else best_model['model']  # Take care of distributed/parallel training

        save_num = 0
        while (save_num < 10):
            try:
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                tokenizer.save_pretrained(output_dir)
                break
            except:
                save_num += 1
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))

    return global_step, tr_loss / global_step

# Function to provide sample-level evaluation
# Used for saving the best models (on validation set) during training
def evaluate(args, model, eval_dataset=None, prefix=""):

    results = []
    t_start = time.time()
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    correct_num = 0

    for batch in eval_dataloader:
        model.eval()

        pair_ids = batch[1]
        batch = tuple(t.to(args.device) for t in batch[0])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3],
                  'img_feats':      batch[4]
            }
            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

            correct = logits.argmax(1) == batch[3].view(-1)
            correct_num += correct.sum().item()

    acc = float(correct_num) / len(eval_dataset)

    logger.info("Eval Results:")
    logger.info("Eval Accuracy: %.3f" % (100*acc))
    logger.info("Eval Loss: %.3f" % (eval_loss))

    t_end = time.time()
    logger.info('Eval Time Cost: %.3f' % (t_end - t_start))

    return results, acc

# Make CLS and KE predictions
# Also does evaluation on structural accuracy
def predict(args, model, eval_dataset=None, tokenizer=None, node_edge_indices=None, tag2region=None, prioritize_cls=False):

    node_edge_indices = copy.deepcopy(node_edge_indices)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, num_workers=args.workers, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Prediction on Test Set *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    predictions = {}
    cls_labels = {}

    struc_total = 0
    struc_correct = 0

    for batch in eval_dataloader:
        model.eval()

        pair_ids, img_len = batch[1], batch[2]
        batch = tuple(t.to(args.device) for t in batch[0])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[3],
                  'img_feats':      batch[4]
            }
            outputs = model(**inputs)

            _, logits = outputs[:2]
            hidden_states = outputs[2]

            for pid_i, pid in enumerate(pair_ids):
                predictions[pid] = {'cls': logits[pid_i, :].argmax().item(), 'ke': {}}
                cls_labels[pid] = inputs['labels'][pid_i].item()

            if args.ke_loss:

                sep_idxs = torch.nonzero(inputs['input_ids'] == tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0])

                for pid_i, pid in enumerate(pair_ids):

                    img_id = pid.split('.')[0]

                    chunk_label = inputs['labels'][pid_i].item()

                    if not node_edge_indices[pid]['node_indices']: 
                        # If no chunks, continue to next example
                        continue

                    node_indices = node_edge_indices[pid]['node_indices']
                    edge_indices = node_edge_indices[pid]['edge_indices']
                    edges = node_edge_indices[pid]['edges']

                    hyp_hidden_states = hidden_states[pid_i, 1:sep_idxs[3*pid_i, 1], :]
                    if args.classifier == 'att+region':
                        tag_hidden_states = hidden_states[pid_i, sep_idxs[3*pid_i, 1]+1:sep_idxs[3*pid_i+1, 1], :]
                        end_img_idx = None if not -args.max_img_seq_length+img_len[pid_i].item() else -args.max_img_seq_length+img_len[pid_i].item()
                        img_hidden_states = hidden_states[pid_i, -args.max_img_seq_length:end_img_idx, :]

                    hyp_len = hyp_hidden_states.size(0)

                    node_feats = []
                    edge_feats = []

                    predictions[pid]['ke'] = {'node': [], 'edge': []}

                    ##### Remove indices that are outside of the sequence
                    for i in range(len(node_indices)):
                        node_indices[i] = [x for x in node_indices[i] if x < hyp_len]
                    for i in range(len(edge_indices)):
                        edge_indices[i] = [x for x in edge_indices[i] if x < hyp_len]
                    for e in list(edges.keys()):
                        if not edge_indices[e]:
                            del edges[e]
                            predictions[pid]['ke']['edge'].append(3) # 3 indicates truncated nodes
                        else:
                            predictions[pid]['ke']['edge'].append(-1) # -1 is a placeholder
                    #####

                    valid_chunks = []
                    for i, chunk in enumerate(node_indices):

                        if not chunk:
                            if args.classifier == 'att+region':
                                node_feats.append(torch.zeros((hyp_hidden_states.size(1)*2)).cuda())
                            else:
                                node_feats.append(torch.zeros_like(hyp_hidden_states[0, :]).cuda())
                            predictions[pid]['ke']['node'].append(3) # 3 indicates truncated nodes
                            continue

                        valid_chunks.append(i)
                        predictions[pid]['ke']['node'].append(-1) # -1 is a placeholder

                        chunk_feat = hyp_hidden_states[chunk, :]
                        alignment_score = model.ke_query(chunk_feat)
                        node_feat = torch.sum(torch.softmax(alignment_score, dim=0)*chunk_feat, dim=0)

                        if args.classifier == 'att+region':
                            if tag_hidden_states.size(0) == 0:
                                obj_feat = img_hidden_states[0]
                            else:
                                salient_node_feat = chunk_feat.detach()[torch.argmax(alignment_score.detach())]
                                sim = F.cosine_similarity(salient_node_feat, tag_hidden_states.detach(), dim=-1)
                                valid_tag_idx = [t_i for t_i in range(sim.size(0)) if tag2region[img_id][t_i] < img_len[pid_i].item()]
                                tag_idx = valid_tag_idx[torch.argmax(sim[valid_tag_idx])]
                                obj_idx = tag2region[img_id][tag_idx]
                                obj_feat = img_hidden_states[obj_idx]
                            node_feats.append(torch.cat([node_feat, obj_feat]))

                        elif args.classifier == 'att':
                            node_feats.append(node_feat)

                    if not valid_chunks:
                        # If no valid chunks, continue to next example
                        continue

                    node_feats = torch.stack(node_feats, dim=0)

                    relations = []
                    for e, n in edges.items():
                        relations.append(node_indices[n[0]] + edge_indices[e] + node_indices[n[1]])

                
                    for chunk in relations:

                        chunk_feat = hyp_hidden_states[chunk, :]
                        alignment_score = model.ke_query(chunk_feat)
                        edge_feat = torch.sum(torch.softmax(alignment_score, dim=0)*chunk_feat, dim=0)

                        if args.classifier == 'att+region':
                            if tag_hidden_states.size(0) == 0:
                                obj_feat = img_hidden_states[0]
                            else:
                                important_edge_feat = chunk_feat.detach()[torch.argmax(alignment_score.detach())]
                                sim = F.cosine_similarity(important_edge_feat, tag_hidden_states.detach(), dim=-1)
                                valid_tag_idx = [t_i for t_i in range(sim.size(0)) if tag2region[img_id][t_i] < img_len[pid_i].item()]
                                tag_idx = valid_tag_idx[torch.argmax(sim[valid_tag_idx])]
                                obj_idx = tag2region[img_id][tag_idx]
                                obj_feat = img_hidden_states[obj_idx]
                            edge_feats.append(torch.cat([edge_feat, obj_feat]))

                        elif args.classifier == 'att':
                            edge_feats.append(edge_feat)

                    if relations:
                        edge_feats = torch.stack(edge_feats, dim=0)
                        node_logits = model.ke_classifier(node_feats)
                        edge_logits = model.ke_classifier(edge_feats)
                        chunk_logits = torch.cat([node_logits[valid_chunks, :], edge_logits], dim=0)
                    else:
                        node_logits = model.ke_classifier(node_feats)
                        chunk_logits = node_logits[valid_chunks, :]

                    chunk_preds = chunk_logits.argmax(dim=1)

                    # node predictions
                    for n_i, pred in enumerate(node_logits.argmax(dim=1).tolist()):
                        if predictions[pid]['ke']['node'][n_i] == 3: 
                            continue
                        else:
                            predictions[pid]['ke']['node'][n_i] = pred

                    # edge predictions
                    if relations:
                        e_i = 0
                        for pred in edge_logits.argmax(dim=1).tolist():
                            while not predictions[pid]['ke']['edge'][e_i] == -1:
                                e_i += 1
                            predictions[pid]['ke']['edge'][e_i] = pred

                        node_preds = node_logits.argmax(dim=1)
                        edge_preds = edge_logits.argmax(dim=1)


                    # Structural Accuracy is calculated below since we have access to the node/edge indices here
                    # Other metrics are calculated in test()

                    for e, n in edges.items():
                        # prioritize CLS predictions (CLS predicts entailment means all KEs are entailed)
                        if prioritize_cls and logits[pid_i, :].argmax().item() == 0:
                            struc_correct += 1
                            struc_total += 1
                            continue

                        #### Top-Down Constraints ####
                        relation_pred = edge_preds[e].item()
                        if relation_pred == 0: 
                            # PARENT IS ENTAILMENT, MUST BE E
                            if node_preds[n[0]].item() == 0 and ((not node_indices[n[1]]) or node_preds[n[1]].item() == 0):
                                struc_correct += 1
                        elif relation_pred == 1: 
                            # PARENT IS NEUTRAL, MUST NOT BE C
                            if node_preds[n[0]].item() != 2 and ((not node_indices[n[1]]) or node_preds[n[1]].item() != 2):
                                struc_correct += 1
                        elif relation_pred == 2: 
                            # PARENT IS CONTRADICTION, NO CONSTRAINT
                            struc_correct += 1
                        struc_total += 1

                        # Bottom-up Constraints
                        for n_i in [0, 1]:
                            if not node_indices[n[n_i]]: continue
                            n_pred = node_preds[n[n_i]].item()
                            if n_pred == 0: 
                                # CHILD IS ENTAILMENT, NO CONSTRAINT
                                struc_correct += 1
                            elif n_pred == 1: 
                                # CHILD IS NEUTRAL, PARENT NOT ENTAILMENT
                                if edge_preds[e].item() != 0: struc_correct += 1
                            elif n_pred == 2: 
                                # CHILD IS CONTRADICTION, PARENT MUST BE C
                                if edge_preds[e].item() == 2: struc_correct += 1
                            struc_total += 1

    return predictions, cls_labels, struc_correct/struc_total

def _clean_node_token(tokens_n):
    if not tokens_n[0].startswith('z'):
        return ''.join(tokens_n).replace('##', '')
    tokens_n = ' '.join(tokens_n).replace(' ##', '').split(' ')
    z = tokens_n[0]
    name = ''.join([t for t in tokens_n[1:] if not t == z])
    ke_str = ' '.join([z, name])
    return ke_str.lower()

# reformat predictions to match the format of the annotation file
def reformat(predictions, node_edge_indices, prioritize_cls=False):
    logger.info('Reformatting predictions to human readable format...')
    readable_predictions = {}
    for pid, pred in predictions.items():
        readable_predictions[pid] = {'cls': pred['cls']}
        example = node_edge_indices[pid]
        if not 'tokens' in example:
            print(pid, example)
        tokens = example['tokens']

        for n_i, n in enumerate(example['node_indices']):
            tokens_n = [tokens[i] for i in n]
            ke_str = _clean_node_token(tokens_n)

            if prioritize_cls and readable_predictions[pid]['cls'] == 0:  # Ours+CLS
                curr_pred = 0
            else:
                curr_pred = pred['ke']['node'][n_i]
            readable_predictions[pid][ke_str] = curr_pred

        for e_i, (n_i1, n_i2) in example['edges'].items():
            n1 = [tokens[i] for i in example['node_indices'][n_i1]]
            n1_str = _clean_node_token(n1)
            n2 = [tokens[i] for i in example['node_indices'][n_i2]]
            n2_str = _clean_node_token(n2)
            edge = ''.join([tokens[i].lower() for i in example['edge_indices'][e_i]]).replace('##', '')
            if edge[-3:] == '-of':
                ke_str = n2_str + ' ' + edge[:-3] + ' ' + n1_str
            else:
                ke_str = n1_str + ' ' + edge + ' ' + n2_str

            if prioritize_cls and readable_predictions[pid]['cls'] == 0:  # Ours+CLS
                curr_pred = 0
            else:
                curr_pred = pred['ke']['edge'][e_i]
            readable_predictions[pid][ke_str] = curr_pred

    return readable_predictions


def test(readable_predictions, cls_labels, fgve_labels, struc_acc):
    # Labels: 0 - entailment, 1 - neutral, 2 -contradiction, 3 - opt-out
    logger.info("***** Running Evaluation on Test Set *****")

    results = {'cls_acc': None, 
                'ke2cls_acc': None, 
                'relab_cls_acc': None, 
                'relab_ke2cls_acc': None, 
                'ke_acc': None, 
                'struc_acc': struc_acc}

    ve_cls_preds = []
    ve_ke2cls_preds = []
    ve_cls_gold = []

    fgve_indices = []

    fgve_cls_gold = []
    fgve_ke_preds = []
    fgve_ke_gold = []

    for pid_i, (pid, predictions) in enumerate(readable_predictions.items()):
        ve_cls_preds.append(predictions['cls'])
        ve_cls_gold.append(cls_labels[pid])
        ve_ke2cls_preds.append(
            max([pred for (ke_str, pred) in predictions.items() if (ke_str != 'cls' and pred != 3)])
        )

        if pid in fgve_labels:
            fgve_indices.append(pid_i)
            for ke_str, label in fgve_labels[pid].items():
                if ke_str == 'hyp':
                    # Sample-level label
                    fgve_cls_gold.append(label)
                else:
                    fgve_ke_preds.append(predictions[ke_str])
                    fgve_ke_gold.append(label)

    ve_cls_preds = np.array(ve_cls_preds, dtype=int)
    ve_ke2cls_preds = np.array(ve_ke2cls_preds, dtype=int)
    ve_cls_gold = np.array(ve_cls_gold, dtype=int)

    fgve_cls_gold = np.array(fgve_cls_gold, dtype=int)
    fgve_ke_preds = np.array(fgve_ke_preds, dtype=int)
    fgve_ke_gold = np.array(fgve_ke_gold, dtype=int)

    # Full VE eval
    # CLS Acc, KE->CLS Acc
    results['cls_acc'] = (ve_cls_preds == ve_cls_gold).sum() / ve_cls_gold.shape[0]
    results['ke2cls_acc'] = (ve_ke2cls_preds == ve_cls_gold).sum() / ve_cls_gold.shape[0] 

    # FGVE eval
    # Relabeled CLS Acc, Relabeled KE->CLS Acc, KE Acc
    results['relab_cls_acc'] = (ve_cls_preds[fgve_indices] == fgve_cls_gold).sum() / fgve_cls_gold.shape[0]
    results['relab_ke2cls_acc'] = (ve_ke2cls_preds[fgve_indices] == fgve_cls_gold).sum() / fgve_cls_gold.shape[0]

    valid_ke_indices = np.where(fgve_ke_gold != 3)[0]
    results['ke_acc'] = (fgve_ke_preds[valid_ke_indices] == fgve_ke_gold[valid_ke_indices]).sum() / len(valid_ke_indices)


    logger.info("***** Eval Results on Full VE Test Set *****")
    logger.info("Number of Samples: %d" % ve_cls_gold.shape[0])
    logger.info("Number of KE->CLS correct: %d" % (ve_ke2cls_preds == ve_cls_gold).sum())
    logger.info("CLS Acc: %.3f" % (results['cls_acc']*100))
    logger.info("KE->CLS Acc: %.3f" % (results['ke2cls_acc']*100))
    logger.info("Structural Acc: %.3f" % (results['struc_acc']*100))

    logger.info("***** Eval Results on FGVE Test Set *****")
    logger.info("Valid Samples: %d" % fgve_cls_gold.shape[0])
    logger.info("Valid KEs: %d" % len(valid_ke_indices))
    logger.info("CLS Acc: %.3f" % (results['relab_cls_acc']*100))
    logger.info("KE->CLS Acc: %.3f" % (results['relab_ke2cls_acc']*100))
    logger.info("KE Acc: %.3f" % (results['ke_acc']*100))

    return results

# Initialize AMR token embeddings
def add_amr_embeddings(args, model, tokenizer, new_tokens, new_special_tokens):
    logger.info('Adding AMR Tokens...')
    substitute_list = json.load(open(os.path.join(args.data_dir, 'amr_substitute.json'), 'r'))
    substitute_list = {v[1:]: k[1:] for k, v in substitute_list.items()}
    logger.info('Substitute List: {}'.format(substitute_list))

    prev_vocab_len = len(tokenizer)

    with torch.no_grad():
        if new_tokens:
            additional_embeddings = torch.zeros((len(new_tokens), model.bert.embeddings.word_embeddings.weight.size(-1)))
            additional_embeddings.normal_(mean=0.0, std=model.bert.config.initializer_range)

            if args.init_new_vocab:
                for i, new_token in enumerate(tqdm(new_tokens)):
                    if new_token.startswith(':'):
                        new_token = new_token[1:]

                        # Reverse substitutes so that we can get the true embeddings
                        if new_token in substitute_list:
                            new_token = substitute_list[new_token]

                        if new_token.startswith('snt'):
                            new_token = 'sentence'
                        elif new_token.lower().startswith('arg'):
                            new_token = 'argument'
                        elif new_token.startswith('op'):
                            new_token = 'operand'

                        if new_token.startswith('prep-'):
                            new_token = new_token[5:]

                        new_token = ' '.join(new_token.split('-'))

                    ids = tokenizer.encode(new_token)
                    additional_embeddings[i, :] = torch.mean(model.bert.embeddings.word_embeddings.weight.data[ids], dim=0)

            tokenizer.add_tokens(new_tokens)
            if len(tokenizer) - prev_vocab_len != len(new_tokens):
                raise RuntimeError("Added tokens number does not match.")
            else:
                model.bert.resize_token_embeddings(len(tokenizer))
                model.bert.embeddings.word_embeddings.weight.data[-len(new_tokens):, :] = additional_embeddings

        tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

    logger.info('Added %d Tokens and %d Special Tokens.' % (len(new_tokens), len(new_special_tokens)))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--img_feature_dir", default=None, type=str, required=True,
                        help="Feature directory.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default='ve', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--loss_type", default='xe', type=str)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run test on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out for BERT.")
    parser.add_argument("--classifier", default='att+region', type=str, help="[arr | att+region]")

    parser.add_argument('--conf_thres', type=float, default=0.2, help="confidence threshold for object detections")
    parser.add_argument("--max_img_seq_length", default=50, type=int, help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='npz', type=str, help="Image feature format (we only support .npz for now).")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=4000, help="Log every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization")
    parser.add_argument('--type_vocab_size', type=int, default=3, help="AMR, object tag, text (in total 3 sequence types). Image tokens do not need token type embeddings")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')

    parser.add_argument("--ke_loss", action='store_true', help="Multi-Instance Learning")
    parser.add_argument("--struc_loss", action='store_true', help="Structural Loss")

    parser.add_argument("--init_new_vocab", action='store_true', help="init new vocab embedding with subword embeddings")

    parser.add_argument("--lr_decay", action='store_true', help="If True, linear decay.")
    parser.add_argument("--prioritize_cls", action='store_true', help="If True, use CLS predictions for all entailment samples' KEs.")

    parser.add_argument('--loss_weights', nargs='+', type=float, default=[0.5, 1.0, 1.0], help='CLS, KE (MIL), Structural Loss weights')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train: logger.info("Output Directory Exists.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args.seed, args.n_gpu)

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name,
    )

    if args.do_train:
        with open(os.path.join(args.data_dir, 'amr_vocab.txt'), 'r') as f:
            new_tokens = f.read().split('\n')
        while '' in new_tokens:
            new_tokens.remove('')

        with open(os.path.join(args.data_dir, 'amr_special_tokens.txt'), 'r') as f:
            new_special_tokens = f.read().split('\n')
        while '' in new_special_tokens:
            new_special_tokens.remove('')

    elif args.do_test:
        checkpoint = None
        for file in os.listdir(args.output_dir):
            if file.startswith('best-'):
                checkpoint = os.path.join(args.output_dir, file)
                break
        if checkpoint is None:
            raise ValueError('No best directory found. Make sure its name starts with \'best-\'.')
        new_tokens = list(json.load(open(os.path.join(checkpoint, 'added_tokens.json'))).values())
        new_special_tokens = json.load(open(os.path.join(checkpoint, 'special_tokens_map.json')))['additional_special_tokens']

    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.classifier = args.classifier
    config.type_vocab_size = args.type_vocab_size
    config.additional_vocab_size = len(new_tokens)
    config.prev_vocab_size = config.vocab_size

    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        add_amr_embeddings(args, model, tokenizer, new_tokens, new_special_tokens)
    elif args.do_test:
        # There was an issue in this version of transformers that cannot load added tokens correctly by tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        # It would show the correct vocab size (len(tokenizer) == tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)) 
        # But it would not use the new tokens during tokenization
        # So we have to reload the initial tokenizer and re-add the AMR tokens to make sure it actually uses them
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        tokenizer.add_tokens(new_tokens)
        tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

    config.vocab_size = len(tokenizer)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.do_train:
        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    print('Loading node edge indices...')
    # Get node and edge chunks
    node_edge_indices = pickle.load(open(os.path.join(args.data_dir, 'node_edge_indices.pkl'), 'rb'))
    print('Loading tag to object token mapping...')
    tag2region = pickle.load(open(os.path.join(args.data_dir, 'tag2region.pkl'), 'rb'))

    # Train
    if args.do_train:
        train_dataset = VEDataset(args, 'train', tokenizer)
        eval_dataset = VEDataset(args, 'val', tokenizer)
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer, 
            node_edge_indices=node_edge_indices, tag2region=tag2region)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]: os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Test
    elif args.do_test and args.local_rank in [-1, 0]:

        fgve_labels = json.load(open(os.path.join(args.data_dir, 'amr_annotations.json')))

        test_dataset = VEDataset(args, 'test', tokenizer)

        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)

        predictions, cls_labels, struc_acc = predict(args, model, test_dataset, tokenizer, 
            node_edge_indices=node_edge_indices, tag2region=tag2region, prioritize_cls=args.prioritize_cls)
        readable_predictions = reformat(predictions, node_edge_indices, prioritize_cls=args.prioritize_cls)

        # Save predictions for convenience
        #json.dump(readable_predictions, open(os.path.join(args.output_dir, 'predictions.json'), 'w'), indent=4)
        #pickle.dump(cls_labels, open(os.path.join(args.output_dir, 'cls_labels.pkl'), 'wb'))

        # Load predictions if already saved
        # readable_predictions = json.load(open(os.path.join(args.output_dir, 'predictions.json'), 'r'))
        # cls_labels = pickle.load(open(os.path.join(args.output_dir, 'cls_labels.pkl'), 'rb'))
        # struc_acc = 0.9636

        results = test(readable_predictions, cls_labels, fgve_labels, struc_acc)
        json.dump(results, open(os.path.join(args.output_dir, 'test_results.json'), 'w'), indent=4) # save test results

if __name__ == "__main__":
    main()
