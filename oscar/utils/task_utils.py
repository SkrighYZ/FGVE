# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
# Copyright (c) 2022 Yipeng Zhang. Licensed under the BSD 3-clause license.
# Modified from https://github.com/microsoft/Oscar

from __future__ import absolute_import, division, print_function

import csv, json
import logging
import os
import sys
from io import open
import _pickle as cPickle
import torch
import numpy as np

logger = logging.getLogger(__name__)


class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, score=None, img_key=None, q_id=None, pair_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.score = score
        self.img_key = img_key
        self.q_id = q_id
        self.pair_id = pair_id


class InputFeat(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, score, img_feat):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.score = score
        self.img_feat = img_feat


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class VEProcessor(DataProcessor):
    """ Processor for the VTE data set. """

    def get_train_examples(self, data_dir, conf_thres, file_name='vte_train.json'):
        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, conf_thres, "train")

    def get_dev_examples(self, data_dir, conf_thres, file_name='vte_dev.json'):
        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, conf_thres, "dev")

    def get_test_examples(self, data_dir, conf_thres, file_name='vte_test.json'):
        lines = json.load(open(os.path.join(data_dir, file_name)))
        return self._create_examples(lines, conf_thres, "test")

    def get_labels(self):
        return ['entailment', 'neutral', 'contradiction']


    def _create_examples(self, lines, conf_thres, set_type):
        """Creates examples for the training and dev sets."""

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, str(i))
            text_a = line['hyp_amr_cleaned']

            num_under_thres = (np.array(line['conf'], dtype=np.float32) < conf_thres).sum()
            text_b = ' '.join(line['objs'].split(' ')) if num_under_thres == 0 else ' '.join(line['objs'].split(' ')[:-num_under_thres])

            text_c = line['hyp']
            label = line['ans']
            img_key = line['feat_id'] if 'feat_id' in line else line['img_id']
            pair_id = line['pair_id']
            score = 0
            q_id = 0
            examples.append(InputInstance(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label, score=score, img_key=img_key, q_id=q_id, pair_id=pair_id))
        
        return examples

def _truncate_seq_pair_ve(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # Truncate tokens_c (hyp text) and tokens_b (object tags) first then tokens_a (hyp AMR)

    if not tokens_b:
        while True:
            total_length = len(tokens_a) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_c) > 0:
                tokens_c.pop()
            else:
                tokens_a.pop()

    else:
        while True:
            total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
            if total_length <= max_length:
                break
            if len(tokens_c) > 0:
                tokens_c.pop()
            elif len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()




processors = {
    "ve": VEProcessor
}

output_modes = {
    "ve": "classification"
}