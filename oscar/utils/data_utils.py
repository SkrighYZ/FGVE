# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
# Copyright (c) 2022 Yipeng Zhang. Licensed under the BDS 3-clause license.
# Modified from https://github.com/microsoft/Oscar

from __future__ import absolute_import, division, print_function

import os
import base64
import copy, time, json

import numpy as np
import torch
from torch.utils.data import Dataset

from .task_utils import _truncate_seq_pair_ve, output_modes, processors

def _load_dataset(args, name):
    processor = processors[args.task_name]()
    labels = processor.get_labels()

    if name == 'train':
        examples = processor.get_train_examples(args.data_dir, args.conf_thres, 've_train.json') 
    elif name == 'val':
        examples = processor.get_dev_examples(args.data_dir, args.conf_thres, 've_dev.json') 
    elif name == 'test': # test-submission
        examples = processor.get_test_examples(args.data_dir, args.conf_thres, 've_test.json')
    else:
        raise ValueError('Data separation is not input correctly.')

    return examples, labels

class VEDataset(Dataset):
    """ VE Dataset """

    def __init__(self, args, name, tokenizer):
        super(VEDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        self.output_mode = output_modes[args.task_name]
        self.tokenizer = tokenizer
        self.args = args
        self.name = name

        self.examples, self.labels = _load_dataset(args, name)
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        print('%s Data Examples: %d' % (name, len(self.examples)))


    def tensorize_example(self, example, cls_token_at_end=False, pad_on_left=False,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                    sequence_a_segment_id=0, sequence_b_segment_id=1, sequence_c_segment_id=2,
                    cls_token_segment_id=1, pad_token_segment_id=0,
                    mask_padding_with_zero=True):

        tokens_a = self.tokenizer.tokenize(example.text_a)

        assert example.text_c is not None # need to modify this if we want to remove hypothesis text

        tokens_b = self.tokenizer.tokenize(example.text_b)
        tokens_c = self.tokenizer.tokenize(example.text_c)
        # Account for [CLS], [SEP], [SEP], [SEP] with "- 4"
        _truncate_seq_pair_ve(tokens_a, tokens_b, tokens_c, self.args.max_seq_length-4)

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        else:
            tokens += [sep_token]
            segment_ids += [sequence_b_segment_id]

        sequence_c_segment_id = sequence_c_segment_id if self.args.type_vocab_size == 3 else sequence_a_segment_id
        
        if tokens_c:
            tokens += tokens_c + [sep_token]
            segment_ids += [sequence_c_segment_id] * (len(tokens_c) + 1)
        else:
            tokens += [sep_token]
            segment_ids += [sequence_c_segment_id]

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(segment_ids) == self.args.max_seq_length



        # image features
        img_feat = self.get_image(example.img_key) # torch
        img_len = min(img_feat.shape[0], self.args.max_img_seq_length)

        if img_feat.shape[0] > self.args.max_img_seq_length:
            img_feat = img_feat[0:self.args.max_img_seq_length, ]
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.args.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.args.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]


        return (torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(input_mask, dtype=torch.long),
                    torch.tensor(segment_ids, dtype=torch.long),
                    torch.tensor(self.label_map[example.label], dtype=torch.long),
                    img_feat,
                    torch.tensor([example.q_id], dtype=torch.long)), example.pair_id, img_len


    def get_image(self, feat_id):
        if self.args.img_feature_type == 'npz':
            data = np.load(os.path.join(self.args.img_feature_dir, feat_id+'.npz'))
            features = data['x']
        else:
            raise ValueError('Image feature type no found.')
        return torch.from_numpy(features)


    def __getitem__(self, index):
        entry = self.examples[index]
        example, pair_id, img_len = self.tensorize_example(entry,
            cls_token_at_end=bool(self.args.model_type in ['xlnet']), # xlnet has a cls token at the end
            cls_token=self.tokenizer.cls_token,
            sep_token=self.tokenizer.sep_token,
            cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(self.args.model_type in ['xlnet']), # pad on the left for xlnet
            pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        return example, pair_id, img_len

    def __len__(self):
        return len(self.examples)