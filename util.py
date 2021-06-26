from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import config 
import os 
import numpy as np
import logging
from collections import defaultdict
import time 
fileHandler = defaultdict(lambda: False)

def getLogger(name):
    if not os.path.exists("log"):
        os.mkdir("log")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)5s] %(filename)s    %(message)s')
    _logger = logging.getLogger(name)
    if not fileHandler[name]:
        uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S.log",time.localtime()) 
        fh = logging.FileHandler(os.path.join('log', uuid_str), mode='a', encoding='utf-8', delay=False)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s][%(levelname)5s] %(filename)s    %(message)s")
        fh.setFormatter(fmt)
        _logger.handlers.append(fh)
        fileHandler[name] = True
    return _logger

class GAIICDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, lines: List[str], mode: int):
        lines1 = [line.split('\t')[0] for line in lines]
        lines2 = [line.split('\t')[1] for line in lines]
        if mode == 1:
            labels = [int(line.split('\t')[2]) for line in lines]

        examples1 = tokenizer(lines1, padding='max_length', truncation=True, max_length=config.max_seq_length,
                              return_tensors='pt', return_special_tokens_mask=True)
        examples2 = tokenizer(lines2, padding='max_length', truncation=True, max_length=config.max_seq_length,
                              return_tensors='pt', return_special_tokens_mask=True)
        examples2['token_type_ids'].fill_(1)
        examples = {key: torch.cat([examples1[key], examples2[key]], 1) for key in examples1}
        if mode == 1:
            examples['labels'] = torch.tensor(labels)
        if mode != 0:
            examples.pop('special_tokens_mask')
        self.examples = [{key: value[i] for key, value in examples.items()} for i in range(len(examples['input_ids']))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, lines: List[str], mode: int):
        '''
        Args：
        mode: 0 for pretrain
            1 for finetune
            2 for test
        '''
        if(mode==0):
            batch_encoding = tokenizer(lines,truncation=True, max_length=config.max_seq_length, return_attention_mask=False,return_token_type_ids=False, padding='max_length', return_tensors='pt')
        else:
            text_a = [line.strip().split('\t')[0] for line in lines]
            text_b = [line.strip().split('\t')[1] for line in lines]
            batch_encoding = tokenizer(text=text_a,text_pair=text_b, truncation=True, max_length=config.max_seq_length, padding='max_length', return_tensors='pt')
            if(mode==1):
                label = [int(line.strip().split('\t')[2]) for line in lines]
                batch_encoding['labels'] = label
                
        self.examples = [{key:value[i] for key,value in batch_encoding.items()} for i in range(len(batch_encoding['input_ids']))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

@dataclass
class DataCollatorForLM2gram:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"],special_tokens_mask)
        return batch

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability / 2)
        special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[:, :-1] |= masked_indices[:, 1:] # TODO: this is 2-gram !!!!
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def mask_ngram(self, inputs: torch.Tensor, max_ngram: int=3, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, max_ngram + 1)
        pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
        

@dataclass
class DataCollatorForLMngram:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    plm_probability: float = 0.15
    max_span_length: int = 3

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])
        return batch

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
            1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be
                masked)
            2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be
                masked
            3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length -
                span_length]`` and mask tokens ``start_index:start_index + span_length``
            4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in
                the sequence to be processed), repeat from Step 1.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see relevant comments in source code for details."
            )

        labels = inputs.clone()
        # Creating the mask and target_mapping tensors
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)

        for i in range(labels.size(0)):
            # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            cur_len = 0
            max_len = labels.size(1)
            while cur_len < max_len:
                ngrams = np.arange(1, self.max_span_length + 1, dtype=np.int64)
                pvals = 1. / np.arange(1, self.max_span_length + 1)
                pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = np.random.choice(ngrams, p=pvals)
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / self.plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
                masked_indices[i, start_index : start_index + span_length] = 1
                # Set `cur_len = cur_len + context_length`
                cur_len += context_length
        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)
        # inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs.long(), labels.long()