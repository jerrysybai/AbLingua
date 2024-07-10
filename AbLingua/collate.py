import argparse
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict, List, Tuple

import torch


class Simple_Collator:
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group(
            "Data Collator Config & Hyperparameter."
        )

        parser.add_argument(
            "--max_len", default=256, type=int
        )  # max length of sequence
        parser.add_argument(
            "--ignore_label", default=-100, type=int
        )  # pytorch standard ignore_label: -100
        parser.add_argument(
            "--split_aa_num", default=3, type=int
        )  # new tokenizer split amino acid number

        parser.add_argument("--truncation", default=True, type=bool)
        parser.add_argument(
            "--truncation_mode", default="cut", type=str, choices=["window", "cut"]
        )

        parser.add_argument("--padding", default=True)
        parser.add_argument("--padding_token", default="[PAD]", type=str)

        return parent_parser

    def __init__(self, tokenizer, args) -> None:
        self.tokenizer = tokenizer  # get the tokenizer
        self.max_len = args.max_len
        self.ignore_label = args.ignore_label
        self.split_aa_num = args.split_aa_num

        # truncation, padding, mask
        assert args.truncation_mode in [
            "window",
            "cut",
        ], "truncate mode must be 'window' or 'cut'."
        self.trunc = args.truncation
        self.trunc_mode = args.truncation_mode

        self.padding = args.padding
        self.padding_token = args.padding_token

    def process_tokens(self, tokens_ids: List[int]) -> Tuple[List[int], List[int]]:
        tokens_labels = [self.ignore_label] * len(tokens_ids)
        return tokens_ids, tokens_labels

    def pad_tokens(
        self, tokens_ids: List[int], tokens_labels: List[str]
    ) -> Tuple[List[int], List[int], List[int]]:
        raw_len = len(tokens_ids)

        len_diff = self.max_len - (raw_len % self.max_len)
        tokens_ids += [self.tokenizer.encode(self.padding_token)] * len_diff
        tokens_labels += [self.ignore_label] * len_diff
        tokens_attn_mask = [1] * raw_len + [0] * len_diff

        return tokens_ids, tokens_labels, tokens_attn_mask

    def trunc_tokens(self, data: list) -> List[list]:
        res = []
        tokens_len = len(data)

        if tokens_len <= self.max_len:
            return [data]

        if self.trunc_mode == "window":
            for i in range(tokens_len - self.max_len + 1):
                res.append(deepcopy(data[i : i + self.max_len]))
        elif self.trunc_mode == "cut":
            for i in range(0, tokens_len, self.max_len):
                res.append(deepcopy(data[i : i + self.max_len]))

        return res

    def seq2data(self, seq: str) -> Tuple[List[int], List[int], List[int]]:
        tokens_ids = self.tokenizer.tokenize(seq)  # 1. tokenize the sequence

        tokens_ids, tokens_labels = self.process_tokens(
            tokens_ids
        )  # 2. joint mask and change tokens and generate labels

        if self.padding is True:
            tokens_ids, tokens_labels, tokens_attn_mask = self.pad_tokens(
                tokens_ids, tokens_labels
            )  # 3. padding seqs

        if self.trunc is True:
            tokens_ids, tokens_labels, tokens_attn_mask = [
                self.trunc_tokens(i)
                for i in [tokens_ids, tokens_labels, tokens_attn_mask]
            ]  # 4. truncate data

        return tokens_ids, tokens_labels, tokens_attn_mask

    def __call__(self, data, HF_dataset: bool = False) -> Dict:
        input_ids, labels, attn_mask = [], [], []

        if HF_dataset is False:
            if isinstance(data, str):
                data = [data]  # process single protein sequence for testing

        for i in data:
            seq = i["seq"] if HF_dataset else i
            tokens_ids, tokens_labels, tokens_attn_mask = self.seq2data(seq)

            input_ids.extend(deepcopy(tokens_ids))
            labels.extend(deepcopy(tokens_labels))
            attn_mask.extend(deepcopy(tokens_attn_mask))

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn_mask),
        }