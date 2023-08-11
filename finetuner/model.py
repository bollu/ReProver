#!/usr/bin/env python3
# Fine tune CodeT5 model on the FStar everest dataset.
from __future__ import absolute_import, division, print_function
import datetime
from typing import *
from loguru import logger
import multiprocessing
from tqdm import tqdm
import sys
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import torch
import numpy as np
import json
import random
import os
import argparse
from transformers.trainer_utils import EvalPrediction
from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    BertConfig, BertForMaskedLM, BertTokenizer,
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
    RobertaConfig, RobertaModel, RobertaTokenizer,
    DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
)
from transformers import AutoModelForSequenceClassification

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# https://huggingface.co/transformers/v3.0.2/model_doc/t5.html#t5forconditionalgeneration
# >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
# 
# >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
# >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
# >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
# >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
# >>> loss, prediction_scores = outputs[:2]
# 
# >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
# >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
# >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
# >>> outputs = model.generate(input_ids)
# 
