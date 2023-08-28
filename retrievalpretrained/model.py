#!/usr/bin/env python3
import difflib
import yaml
import numpy as np
from torch import nn
import torch
import argparse
import sys
from dataclasses import dataclass, asdict, field
from typing import *
import os
import random
from copy import deepcopy
import openai
import json
import pandas as pd
import tiktoken
from pathlib import Path
import gzip, pickle
from tqdm import tqdm
from typing import *
from loguru import logger
from openai.embeddings_utils import get_embeddings
import networkx as nx
import torch
import itertools
import pathlib
import inspect
from retrievalfstar.datamodule import Corpus
from torch.utils.data import Dataset, DataLoader
import wandb
import pretty_errors
from retrievalpretrained.sinequanon import SineQuaNon
from retrievalpretrained.mepo import MePo

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# encoder = tiktoken.get_encoding(embedding_encoding)

# openai.organization = os.getenv("OPENAI_ORGANIZATION") #  "org-BkKFe6jDEL6tnBM8uzo3q0MC"
openai.api_key = os.getenv("OPENAI_EMBEDDINGS_API_KEY")
OPENAI_EMBEDDING_VEC_LEN = 1536
logger.debug(f"openAI org: '{openai.organization}' | API key: '{openai.api_key}'")
logger.debug("openAI models:")
openai.Model.list()
logger.debug("-----")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

OPENAI_NSKIPPED : int = 0
def openai_batch(strings :  List[str], max_tokens : int = 3000, batch_size : int = 16):
    """
    batch the list of strings according to openAI batching rules,
    which is that each batch can have at most 5000 tokens,
    and a batch size of at most 2000. Returns the indexes
    of the strings that were skipped as they are too long.
    TODO: consider giving this information eagerly.
    """
    global OPENAI_NSKIPPED
    # actual max tokens is 8192
    skipped = []
    ix = 0
    while ix < len(strings):
        batch = []
        ntokens = 0
        while ix < len(strings) and len(batch) < batch_size:

            if len(strings[ix]) >= max_tokens:
                OPENAI_NSKIPPED += 1
                logger.error("string too long, skipping string at index '{ix}'")
                ix += 1 

            if ntokens + len(strings[ix]) >= max_tokens:
                # batch doesn't have enough space.
                break 

            logger.info(f"building batch, ix={ix}, ntokens={ntokens}")
            ntokens += len(strings[ix]) #upper bound for actual number of tokens.
            batch.append(strings[ix])
            ix += 1

        logger.debug(f"yielding batch")
        if batch == []:
            continue # return back to the top if batch is insufficient.
        yield batch


def aos_to_soa(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """collate and return a list of dicts into a dict of lists"""
    batch = dict()
    # Copy the rest of the fields.
    for k in examples[0].keys():
        batch[k] = [ex[k] for ex in examples]
    return batch


class PretrainedDataset(Dataset):
    data_path : str 
    corpus : Corpus 
    embedded_strings : List[str] # names for which we have embeddings
    num_negatives : int
    is_train : bool
    data : List[Dict[str, Any]]
    def __init__(
        self,
        data_paths: List[str],
        corpus : Corpus,
        is_train: bool,
        num_negatives : int,
        num_in_file_negatives : int,
        embedded_strings : List[str]
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.is_train = is_train
        assert len(data_paths) == 1
        data_path = data_paths[0]
        self.data_path = data_path
        self.embedded_strings = embedded_strings
        self.data = PretrainedDataset._load_data(corpus=self.corpus, data_path=data_paths[0], embedded_strings=self.embedded_strings)
        # self.data = list(
        #     itertools.chain.from_iterable(
        #         PretrainedDataset._load_data(corpus=self.corpus, data_path=path) for path in data_paths
        #     )
        # )

        
    def __repr__(self):
        return f"{type(self)} from path {self.data_path}"

    @staticmethod
    def load_json_or_jsonl(path : str) -> List[Dict[str, Any]]:
        ext = pathlib.Path(path).suffix
        if ".json" == ext:
            return json.load(open(path))
        else:
            assert ".jsonl" == ext
            return [json.loads(line) for line in open(path)]

    def file_path_to_file_name(path : str) -> str: 
        """
        the file import graph only stores file names. So filter file paths into their 
        names (without suffix) so we can correctly compute import inclusion
        """
        return pathlib.path(path).stem

    # TODO: make this @staticmethod.
    @staticmethod
    def _load_data(corpus : Corpus, data_path: str, embedded_strings: List[str]) -> List[Dict[str, Any]]:
        out_data = []
        logger.info(f"Loading data from '{data_path}'")
        in_data = PretrainedDataset.load_json_or_jsonl(data_path)
        for (i, datum) in tqdm(enumerate(in_data)):
            if i == 0:
                logger.info(f"loading datum with keys '{datum.keys()}'")
            # logger.info(f"loading '{i}'th data from '{data_path}'. keys: '{datum.keys()}'")
            if not corpus.has_definition_for_name(datum["name"]) or datum["name"] not in corpus.all_names():
                logger.warning(f"skipping defn '{datum['name']}' as we do not have definition")
                continue

            ctx_def = corpus.get_ctx_embed_str_for_name(datum["name"])
            if ctx_def not in embedded_strings:
                logger.warning(f"skipping defn '{datum['name']}' as we do not have embedding")
                continue

            # TODO: refactor this to be way flatter.
            # context = Context(name=datum["name"],
            #     type_=datum["type"],
            #     definition=datum["definition"])
            # all premises that are *used* in the tactic.
            # only keep those premise names that don't occur in the ctx embedding.
            all_pos_premise_names = [p for p in datum["premises"] if p not in ctx_def]
            if  len(all_pos_premise_names) == 0:
                continue # skip this def cause it has zero premises

            skip = False
            for p in all_pos_premise_names:
                if p not in corpus.all_names(): 
                    skip = True; 
                    break;
                if not corpus.has_definition_for_name(p):
                    skip = True;
                    break;
                if not corpus.get_premise_embed_str_for_name(p) in embedded_strings:
                    skip = True;
                    break
            if skip: continue

            for pos_premise_name in all_pos_premise_names:
                # check that the premise occurs earlier than the context.
                if not corpus.occurs_before_loc_by_name(datum["name"], pos_premise_name):
                    logger.error(f"premise '{pos_premise_name}' in file '{corpus.get_loc_for_name(pos_premise_name)}'")
                    logger.error(f"^^ does not occur before context '{datum['name']}' in file '{corpus.get_loc_for_name(datum['name'])}'")
                assert corpus.occurs_before_loc_by_name(datum["name"], pos_premise_name)
                out_data.append({
                    "context_name": datum["name"],
                    "pos_premise_name": pos_premise_name,
                    "all_pos_premise_names": all_pos_premise_names,
                })

        logger.info(f"Loaded '{len(out_data)}' examples.")
        return out_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Sequence) -> Sequence[Dict[str, Any]]:
        """
        enrich `self.data` with negative samples and return a datum.
        The way this is written, we get _different_ negative sample
        for the same positive sample on different epochs. This makes us
        extra clever.
        Note that if `idx` isa slice, we need to handle this correctly!
        """
        if not self.is_train:
            return self.data[idx]
        # In-file negatives + random negatives from all accessible premises.
        logger.info(f"idx: {idx} | type: {type(idx)} | is int: {isinstance(idx, int)}")
        if isinstance(idx, int):
            exs = [self.data[idx]]
        else:
            exs = self.data[idx]
        exs = deepcopy(exs)
        for ex in exs:
            premises_in_file = []
            premises_outside_file = []
            cur_file_name = self.corpus.get_loc_for_name(ex["context_name"]).file_name
            # collect all negative samples (TODO, HACK: for now, code is disabled).
            if not premises_outside_file:
                   premises_outside_file = self.corpus.all_premise_names # HACK: just store all premise names
                   logger.error(f"unable to find premise outside file '{cur_file_name}'!")

            neg_premise_names = random.sample(premises_outside_file, self.num_negatives)
            # only keep those negative embeddings which have emdataset/QUIC.Spec.Lemmas.jsonbeddings fo them.
            neg_premise_names = [n for n in neg_premise_names if self.corpus.get_premise_embed_str_for_name(n) in self.embedded_strings]
            ex["neg_premise_names"] = neg_premise_names
        if isinstance(idx, int):
            return exs[0]
        else:
            return exs




def soa_dict_microbatch_iter(d : Dict[str, Any], batch_size : int):
    """
    for a dict in structure of arrays form, ie it has keys  where each value is
    a list of the same length, create a list of dicts, where each new dict
    has entries of size batch_size
    """
    ks = d.keys()
    k0 = ks[0]
    n = len(d[k0])
    for ix in range(0, n, batch_size):
        batch = { k : d[k][ix:ix+batch_size] for k in ks }
        yield batch

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, ndim_in : int, ndim_hidden : int, ndim_out : int):
    super().__init__()
    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(ndim_in, ndim_hidden),
      nn.ReLU(),
      nn.Linear(ndim_hidden, ndim_hidden),
      nn.ReLU(),
      nn.Linear(ndim_hidden, ndim_out)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MyModel:
    # indexed by string index.
    corpus : Corpus
    train_dataset : PretrainedDataset
    validate_dataset: PretrainedDataset
    embeddings : Dict[str, torch.tensor]
    minibatch_size : int
    microbatch_size : int
    nepoch : int 
    embedded_strings : List[str] # names for which we have embeddings
    ctx_projection : nn.Module
    premise_projection : nn.Module
    on_epoch_end_callbacks : List[Callable] # callback args: (model=MyModel, epoch_loss=epoch_los)
    optimizer : torch.optim.Optimizer
    lrscheduler : torch.optim.lr_scheduler.LRScheduler

    def __init__(self,
                 corpus : Corpus,
                 embeddings : Dict[str, torch.tensor], 
                 train_dataset : PretrainedDataset,
                 validate_dataset : PretrainedDataset,
                 nepoch : int,
                 minibatch_size : int,
                 microbatch_size : int,
                 embedded_strings : List[str]): 
        self.embedded_strings = embedded_strings
        self.corpus = corpus
        self.embeddings = embeddings
        self.nepoch = nepoch
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.minibatch_size = minibatch_size
        self.microbatch_size = microbatch_size
        OUR_EMBEDDING_VEC_LEN = OPENAI_EMBEDDING_VEC_LEN
        # self.ctx_projection = MLP(OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN, OUR_EMBEDDING_VEC_LEN)
        # self.premise_projection = MLP(OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN, OUR_EMBEDDING_VEC_LEN)

        # identity operators:
        self.ctx_projection = torch.nn.Linear(OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN)
        self.premise_projection = self.ctx_projection
        # self.ctx_projection = torch.nn.Identity()
        # self.premise_projection = torch.nn.Identity()


        self.on_epoch_end_callbacks = []
        self.optimizer = None
        self.lrscheduler = None

    def to(self, device):
        self.ctx_projection.to(device)
        self.premise_projection.to(device)
        for name in self.embeddings:
            self.embeddings[name] = self.embeddings[name].to(device)

    def parameters(self):
        return list(self.ctx_projection.parameters()) + list(self.premise_projection.parameters())

    def mk_pickle_dict(self):
        return { "ctx_projection" : self.ctx_projection.state_dict(), "premise_projection" : self.premise_projection.state_dict() }

    def load_pickle_dict(self, d):
        self.ctx_projection.load_state_dict(d["ctx_projection"])
        self.premise_projection.load_state_dict(d["premise_projection"])
        self.ctx_projection = self.ctx_projection.to(device)
        self.premise_projection = self.premise_projection.to(device)

    def train_microbatch(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        ctx_names = [d["context_name"] for d in data]
        premise_names = [d["pos_premise_name"] for d in data]

        for d in data:
            premise_names.extend(d["neg_premise_names"])

        # 5 * 3 = 15
        # label[ic, ip] = ctx[i] @ (if ip == 0 then premise[i] else neg_sample[i][...)
        logger.info(f"ctx_names: {len(ctx_names)} | premise_names: {len(premise_names)}")
        labels = []
        ctxs = [] # NCTX x DOPENAI_EMBED
        premises = [] # NPREMISES x DOPENAI_EMBED
        # logger.info(f"known embeds: {self.embeddings.keys()}")
        for (ic, c) in enumerate(ctx_names):
            assert c in self.corpus.all_names()
            ctxs.append(self.embeddings[self.corpus.get_ctx_embed_str_for_name(c)].reshape(1, -1))
        for (ip, p) in enumerate(premise_names):
            assert p in self.corpus.all_names()
            premises.append(self.embeddings[self.corpus.get_premise_embed_str_for_name(p)].reshape(1, -1))

        for (ic, c) in enumerate(ctx_names):
            labels.append([])
            for (ip, p) in enumerate(premise_names):
                is_positive =  p in data[ic]["all_pos_premise_names"]
                labels[ic].append(float(is_positive))
            labels[ic] = torch.tensor(labels[ic]).reshape(1, -1)

        ctxs = torch.cat(ctxs, axis=0).to(device)
        premises = torch.cat(premises, axis=0).to(device)
        # [NCTX x NPREMISE]
        labels = torch.cat(labels, axis=0).to(device)

        logger.info(f"ctxs: {ctxs.shape} | premises: {premises.shape} | labels: {labels.shape}")
        logger.info(f"labels: {labels.shape}")
        print(labels)
        # | ctx_embed: {ctx_embed.shape} | premise_embed: {premise_embed.shape} | dots: {dots.shape}")

        # [NCTX x DEMBED]
        ctx_embed = self.ctx_projection(ctxs)
        # dim = 1 means that we take all of 'dim=1' and normalize, fixing all other indexes.
        ctx_embed = torch.nn.functional.normalize(ctx_embed, dim=1)
        # [NPREMSE x DEMBED]
        premise_embed = self.premise_projection(premises)
        premise_embed = torch.nn.functional.normalize(premise_embed, dim=1)
        logger.info(f"ctxs_embed {ctx_embed.shape} | premise_embed: {premise_embed.shape}")
        # [NCTX x NPREMISE]
        dots = ctx_embed @ premise_embed.T
        logger.info(f"dots: {dots.shape}")
        logger.info(dots)

        assert torch.all(torch.le(dots, torch.Tensor([1.1]).to(device)))
        assert torch.all(torch.ge(dots, torch.Tensor([-1.1]).to(device)))
        logger.info(f"dots: {dots.shape} | labels: {labels.shape}")
        assert dots.shape == labels.shape
        loss = torch.sum(torch.nn.functional.mse_loss(dots, labels))
        return loss

    def train_minibatch(self, records: List[Dict[str, Any]]) -> float:
        self.optimizer.zero_grad()
        loss = 0
        for i in range(0, len(records), self.microbatch_size):
            loss += self.train_microbatch(records[i:i+self.microbatch_size])
        loss.backward()
        loss_val = loss.item()
        self.optimizer.step()
        del loss
        return loss_val

    def train_epoch(self, iepoch: int, records: List[Dict[str, Any]]) -> float:
        epoch_loss = 0
        for i in range(0, len(records), self.minibatch_size):
            logger.info(f".. ..minibatch[{i}:{i+self.minibatch_size}]")
            batch_loss = self.train_minibatch(records[i:i+self.minibatch_size])
            wandb.log({"batch_loss": batch_loss})
            epoch_loss += batch_loss
            logger.info(f"epoch[0-1] {iepoch/self.nepoch:4.2f}, batch[0-1] {i/len(records):4.2f}, batch loss: {batch_loss:5.3f}, epoch_loss: {epoch_loss:5.3f}")
            # self.lrscheduler.step(iepoch + i / len(records))
            self.lrscheduler.step()
        # logger.info(f"epoch loss: {loss.item()}")
        return epoch_loss

    def train(self):
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        self.optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-3)
        # self.lrscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 1, T_mult=2)
        # self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        for e in range(self.nepoch):
            logger.info(f"..epoch: {e}/{self.nepoch}")
            epoch_loss = self.train_epoch(e, self.train_dataset)
            wandb.log({"epoch_loss": epoch_loss})
            logger.info(f"..epoch end: {e}/{self.nepoch}, loss: {epoch_loss}")
            examples, validation = self.test()
            for callback in self.on_epoch_end_callbacks:
                callback(model=self, epoch=e, epoch_loss=epoch_loss, examples=examples, validation=validation)
            logger.info("testing at end of epoch {e}")
            wandb.log(validation)


    def test(self) -> (Dict[str, Any], Dict[str, Any]):
        """
        return a tuple, consisting of (a) the examples, and (b) the final validation losses.
        """
        logger.info(f"running evaluation on {self.validate_dataset}")
        collator = TestStatisticsCollator()

        ix2premise_name = [name for name in self.corpus.all_premise_names if self.corpus.get_premise_embed_str_for_name(name) in self.embedded_strings]
        ix2premise = [self.corpus.get_premise_embed_str_for_name(name) for name in ix2premise_name]
        premise2ix = { ix2premise[ix] : ix for ix in range(len(ix2premise)) }
        # NBATCH x OPENAI_EMBED_DIM
        all_premises_embeds = torch.cat([self.embeddings[ix2premise[ix]].view(1, -1) for ix in range(len(ix2premise)) ], axis=0).to(device)
        # NBATCH x OUR_EMBED_DIM
        all_premises_embeds = self.premise_projection(all_premises_embeds)
        all_premises_embeds = torch.nn.functional.normalize(all_premises_embeds, dim=1)
        logger.info("all_premises_embeds: {all_premises_embeds}")
        # premises indexed by ix.
        examples = []

        seen_contexts = set()
        for record in tqdm(self.validate_dataset, desc=f"evaluation on {self.validate_dataset}"):
            if record["context_name"] in seen_contexts: continue # skip repeated contexts.
            # 1 x OPENAI_EMBED_DIM
            ctx_embed = self.embeddings[self.corpus.get_ctx_embed_str_for_name(record["context_name"])].to(device)
            # 1 x OUR_EMBED_DIM
            ctx_embed = self.ctx_projection(ctx_embed)
            ctx_embed = torch.nn.functional.normalize(ctx_embed, dim=0) # has only a single dimension.
            # OUR_EMBED_DIM
            ctx_embed = ctx_embed.view(-1)
            # OUR_EMBED_DIM x [NBATCH x OUR_EMBED_DIM].T
            # OUR_EMBED_DIM x [OUR_EMBED_DIM; NBATCH]
            similarities = ctx_embed @ all_premises_embeds.T

            logger.info(f"similarities: {similarities.shape}")
            logger.info(similarities)
            assert torch.all(torch.le(similarities, torch.Tensor([1.]).to(device)))
            assert torch.all(torch.ge(similarities, torch.Tensor([-1.]).to(device)))

            ix2rank = similarities.argsort(dim=0, descending=True).tolist()
            all_pos_premise_names = record["all_pos_premise_names"]
            all_pos_premise_names_set = set(all_pos_premise_names)
            retreived_premise_names = [ix2premise_name[ix2rank[i]] for i in range(len(ix2rank))]
            # TODO: filter by accessible premises.
            TP1 = retreived_premise_names[0] in all_pos_premise_names
            R1 = float(TP1) / 1
            collator.R1s.append(R1)
            TP10 = len(all_pos_premise_names_set.intersection(retreived_premise_names[:10]))
            R10 = float(TP10) / len(all_pos_premise_names[:10])
            collator.R10s.append(R10)

            RR = 0
            for j, p in enumerate(retreived_premise_names):
                if p in all_pos_premise_names:
                    RR = (1.0 / (j + 1))
                    break
            collator.RRs.append(RR)

            # AP = integral_0^1 P(r) dr
            # change of variables into k 
            # let rel(k) = 1 if retreived_premises[k] in all_pos_premises else 0
            # let s(k) = sum_{j=0}^k rel(j)
            # p(k) = s(k) / k
            # r = s(k) / |all_pos_premises|
            # dk = 1
            # dr = (r(k + dk) -  r(k)) / dk 
            #    = (r(k + 1) - r(k)) / 1
            #    = rel(k+1) / |all_pos_premises|
            # AP = integral_0^1 P(r) dr
            #    = sum_0^N P(r(k)) dr(k)
            #    = sum_0^N p(k) rel(k) / |all_pos_premises|
            AP = 0
            DCG = 0
            IDCG = 0
            
            K_at_full_recall = np.nan # How to correctly initialize?
            K_percent_at_full_recall = np.nan
            ncorrect_at_j = 0

            for j, p in enumerate(retreived_premise_names):
                discount = np.log2(j + 1) if j > 0 else 1
                if j < len(all_pos_premise_names):
                    IDCG += 1.0 / discount      

                rel_at_j = int(p in all_pos_premise_names)
                if rel_at_j:
                    print(f'ctx: {record["context_name"]:20s} relevant premise: {p:20s}@{j:4d}')
                ncorrect_at_j += rel_at_j

                if ncorrect_at_j == len(all_pos_premise_names):
                    K_at_full_recall = j + 1
                    K_percent_at_full_recall = K_at_full_recall / len(retreived_premise_names)

                DCG += rel_at_j / discount
                p_at_j = ncorrect_at_j / (j + 1)
                AP += p_at_j * rel_at_j
            print(f'AP for {record["context_name"]:20s}: {AP:6f} | #pospremises: {all_pos_premise_names}')
            collator.APs.append(AP)
            NDCG = DCG / IDCG
            collator.NDCGs.append(NDCG)
            collator.Ks_at_full_recall.append(K_at_full_recall)
            collator.Ks_percent_at_full_recall.append(K_percent_at_full_recall)

            examples.append({
                "context_name": record["context_name"],
                "all_pos_premise_names": record["all_pos_premise_names"],
                "retreived_premise_names": retreived_premise_names[:30], # only keep 30, otherwise this gets too large.
                "TP1" : TP1,
                "TP10": TP10,
                "R10" : R10,
                "R1" : R1,
                "NDCG": NDCG,
                "RR" : RR,
                "AP": AP
            })

        # finish processing all records.
        # average NDCG
        R1 = np.mean(collator.R1s)
        R10 = np.mean(collator.R10s)
        MAP = np.mean(collator.APs)
        NDCG = np.mean(collator.NDCGs)
        RR = np.mean(collator.RRs)
        logger.info(f"** Eval on {self.validate_dataset} | R1[0-1]: {R1} , RR[0-1]: {RR}, R10[0-1]: {R10}, MAP[0-1]: {MAP}, NDCG[0-1]: {NDCG} **")
        collated = {"R1": R1, "R10": R10, "MAP": MAP, "NDCG": NDCG, "RR" : RR}
        return examples, collated

def download(corpus_path : str, import_graph_path : str, embeds_path : str):
    corpus  = Corpus(corpus_path, import_graph_path)
    strings = set() # is this deterministic when converting to list?

    logger.info(f"checking if '{embeds_path}' exists...")
    try:
        with gzip.open(embeds_path, "rb") as f:
            embeds = pickle.load(f)
            logger.info(f"{embeds_path} does exist.")
            return # already have embeds.
    except Exception as e:
        logger.error(f"ERROR: failed to load pickle at '{embeds_path}'. {e}")
        logger.error(f"ERROR: unable to load embeds '{embeds_path}'. Recreating embedss...")

    embeds = dict() # Dict[str, torch.tensor]
    for name in tqdm(corpus.all_names(), desc="querying openAI"):
        logger.info(f"processing '{name}'")
        ctx = corpus.get_ctx_embed_str_for_name(name)
        premise = corpus.get_premise_embed_str_for_name(name)
        strings.add(ctx)
        strings.add(premise)

    logger.debug("done querying. Now building gzip record")
    strings = sorted(list(strings)) # is this deterministic? probably not.
    for strings_batch in tqdm(list(openai_batch(strings)), desc="building openAI embeddings"):
        assert len(strings_batch)
        # logger.info(f"vvbatch size: {len(strings_batch)}vv")
        # logger.info(f"\n".join([f"  - '{s}'" for s in strings_batch]))
        # logger.info(f"--")
        vecs = get_embeddings(strings_batch, engine=embedding_model)
        # logger.info(f"vecs: {vecs}")
        assert len(strings_batch) == len(vecs)
        for (k, v) in zip(strings_batch, vecs):
            embeds[k] = torch.tensor(v)

    for (i, k) in enumerate(embeds.keys()):
        logger.info(f"embeds[{k}] =shape={embeds[k].shape}")
        if i >= 3: break

    logger.debug(f"dumping embeds at '{embeds_path}'.")
    os.makedirs(pathlib.Path(embeds_path).parent, exist_ok=True)
    with gzip.open(embeds_path, "w") as f:
        pickle.dump({ "embeds": embeds }, f)
    logger.debug(f"dumped pickle at '{embeds_path}'!")

@dataclass
class TestStatisticsCollator:
    R1s : List[int] = field(default_factory=list)
    test_step_outputs : List[Dict[str, Any]] = field(default_factory=list)
    Ks_at_full_recall : List[int] = field(default_factory=list)
    Ks_percent_at_full_recall : List[int] = field(default_factory=list)
    R10s : List[int] = field(default_factory=list)
    RRs : List[int]= field(default_factory=list)
    APs : List[int] = field(default_factory=list)
    NDCGs : List[int] = field(default_factory=list)

def test(ckpt_path : str, data_path : str, eval_batch_size : int):
    ckpt_path = pathlib.Path(ckpt_path)
    logger.debug("loading pickle...")
    with gzip.open(ckpt_path, "rb") as f:
        loaded = pickle.load(f)
        model = loaded["model"]
    assert model is not None
    assert all_dl is not None
    # assert isinstance(model, Model)
    # assert isinstance(all_dl, DataLoader)
    # logger.info(f"loaded pickle! model #strings: '{len(model.strings)}'")

class ModelSaver:
    model_dir : pathlib.Path
    prev_optimal : float
    def __init__(self, model_dir : str):
        self.model_dir = pathlib.Path(model_dir)
        self.prev_optimal = -np.inf 

    def __call__(self, epoch : int, model: MyModel, epoch_loss : float, examples : Dict[str, Any], validation : Dict[str, Any]):
        os.makedirs(self.model_dir, exist_ok=True)
        cur_val = validation["NDCG"]
        is_cur_best = cur_val > self.prev_optimal  # NDCG should _increase_ not _decrease_, jesus.
        if is_cur_best: 
            prev_optimal = self.prev_optimal
            self.prev_optimal = cur_val
            save_path = self.model_dir / "best.pickle.gz"
            logger.info(f"***saving best model (cur: '{cur_val:4.2f}' < prev: '{prev_optimal:4.2f}') to '{save_path}' ***")
            with gzip.open(save_path, "w") as f:
                pickle.dump(model.mk_pickle_dict(), f)

            record_path = self.model_dir / "best.record.json"
            with open(record_path, "w") as f:
                json.dump({ "examples": examples, "validation": validation}, f, indent=1)

        save_path = self.model_dir / f"epoch-{epoch}.pickle.gz"
        logger.info(f"***saving model @ epoch '{epoch}' to '{save_path}'***")
        with gzip.open(save_path, "wb") as f:
            pickle.dump(model.mk_pickle_dict(), f)

        record_path = self.model_dir / f"epoch-{epoch}.record.json"
        with open(record_path, "w") as f:
            json.dump({ "examples": examples, "validation": validation}, f, indent=1)

def fit(model_dir : str, 
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
    project="premise-selection-code-embeddings-pretrained",
    # Track hyperparameters and run metadata
    config={
        "model_dir": model_dir,
        "corpus_path": corpus_path,
        "import_graph_path":import_graph_path,
        "embeds_path": embeds_path,
        "data_dir": data_dir,
        "minibatch_size": minibatch_size,
        "microbatch_size":microbatch_size
    })


    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{embeds_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]

    assert embeds is not None
    embedded_strings = list(embeds.keys())
    logger.info(f"load embeds. {len(embedded_strings)}")

    logger.info(f"starting model training from path '{data_dir}'")
    model = MyModel(corpus=corpus,
                  embeddings=embeds,
                  minibatch_size=minibatch_size,
                  microbatch_size=microbatch_size,
                  nepoch=nepoch,
                  embedded_strings=embedded_strings,
                  train_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "train.json"],
                                                  corpus=corpus,
                                                  is_train=True,
                                                  num_negatives=1, num_in_file_negatives=1,
                                                  embedded_strings=embedded_strings),
                  validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0,
                                                     num_in_file_negatives=0,
                                                    embedded_strings=embedded_strings))
    model.to(device)
    logger.info(f"training model...")
    os.makedirs(model_dir, exist_ok=True)
    model.on_epoch_end_callbacks.append(ModelSaver(model_dir))
    model.train()


    model_path = pathlib.path(model_dir) / "final.pickle.gz"
    logger.info(f"writing model to {model_path}")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model.mk_pickle_dict(), f)


def test(test_model_path : str, 
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{embeds_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]
    assert embeds is not None
    embedded_strings = list(embeds.keys())
    logger.info(f"load embeds. {len(embedded_strings)}")

    logger.info(f"starting model testing from path '{data_dir}'")
    assert embeds is not None
    model = MyModel(corpus=corpus,
                  embeddings=embeds,
                  minibatch_size=minibatch_size,
                  microbatch_size=microbatch_size,
                  nepoch=nepoch,
                  embedded_strings=embedded_strings,
                  train_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "train.json"],
                                                  corpus=corpus,
                                                  is_train=True,
                                                  num_negatives=2, num_in_file_negatives=1,
                                                  embedded_strings=embedded_strings),
                  validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0,
                                                     num_in_file_negatives=0,
                                                    embedded_strings=embedded_strings))
    logger.info(f"loading model weights from {test_model_path}...")
    with gzip.open(test_model_path, "rb") as f:
        model.load_pickle_dict(pickle.load(f))
    logger.info(f"loaded model.")
    logger.info(f"testing model...")
    model.test()



def sine_qua_non(test_model_path : str, 
        model_dir : str,
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):

    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{embeds_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]
    assert embeds is not None
    embedded_strings = list(embeds.keys())
    logger.info(f"load embeds. {len(embedded_strings)}")

    logger.info(f"starting model testing from path '{data_dir}'")
    assert embeds is not None

    validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0,
                                                     num_in_file_negatives=0,
                                                    embedded_strings=embedded_strings)


    # TODO: don't throw away stuff, just truncate.
    # version of model.test() that implements sine selection
    ix2premise_name = [name for name in corpus.all_premise_names \
                       if corpus.get_premise_embed_str_for_name(name) in embedded_strings]
    ix2premise = [corpus.get_premise_embed_str_for_name(name) for name in ix2premise_name]

    # what are the precise rules here? What do we choose to be the axioms? I guess the full premise DB
    sqn = SineQuaNon(symbols=ix2premise_name, axioms=dict(zip(ix2premise_name, ix2premise)))
    premise2ix = { ix2premise[ix] : ix for ix in range(len(ix2premise)) }

    seen_contexts = set()
    collator = TestStatisticsCollator()
    examples = []
    for record in tqdm(validate_dataset, desc=f"evaluation on {validate_dataset}"):
        if record["context_name"] in seen_contexts: continue # skip repeated contexts.
        full_recall_at_lvl = 0
        cumulative_num_recalled = 0
        cumulative_seen_premises = set()
        level2names = []
        rank2name = []
        level2recall = []
        level2cumulative_seen = []
        for lvl in sqn.goal2selection(corpus.get_ctx_embed_str_for_name(record["context_name"])):
            level2names.append(lvl)
            rank2name.extend(lvl)
            for name in lvl:
                if name in record["all_pos_premise_names"] and name not in cumulative_seen_premises:
                    cumulative_seen_premises.add(name)
                    cumulative_num_recalled += 1
            level2recall.append(cumulative_num_recalled / len(record["all_pos_premise_names"]))
            level2cumulative_seen.append(cumulative_num_recalled)
            # stop processing once we have seen everything we care about.
            if cumulative_num_recalled == len(record["all_pos_premise_names"]):
                break

        retreived_premise_names = rank2name
        all_pos_premise_names = record["all_pos_premise_names"]
        all_pos_premise_names_set = set(all_pos_premise_names)
        # TODO: filter by accessible premises.
        TP1 = retreived_premise_names[0] in all_pos_premise_names
        R1 = float(TP1) / 1
        collator.R1s.append(R1)
        TP10 = len(all_pos_premise_names_set.intersection(retreived_premise_names[:10]))
        R10 = float(TP10) / len(all_pos_premise_names[:10])
        collator.R10s.append(R10)

        RR = 0
        for j, p in enumerate(retreived_premise_names):
            if p in all_pos_premise_names:
                RR = (1.0 / (j + 1))
                break
        collator.RRs.append(RR)

        # AP = integral_0^1 P(r) dr
        # change of variables into k 
        # let rel(k) = 1 if retreived_premises[k] in all_pos_premises else 0
        # let s(k) = sum_{j=0}^k rel(j)
        # p(k) = s(k) / k
        # r = s(k) / |all_pos_premises|
        # dk = 1
        # dr = (r(k + dk) -  r(k)) / dk 
        #    = (r(k + 1) - r(k)) / 1
        #    = rel(k+1) / |all_pos_premises|
        # AP = integral_0^1 P(r) dr
        #    = sum_0^N P(r(k)) dr(k)
        #    = sum_0^N p(k) rel(k) / |all_pos_premises|
        AP = 0
        DCG = 0
        IDCG = 0
        
        K_at_full_recall = np.nan # How to correctly initialize?
        K_percent_at_full_recall = np.nan
        ncorrect_at_j = 0

        for j, p in enumerate(retreived_premise_names):
            discount = np.log2(j + 1) if j > 0 else 1
            if j < len(all_pos_premise_names):
                IDCG += 1.0 / discount      

            rel_at_j = int(p in all_pos_premise_names)
            if rel_at_j:
                print(f'ctx: {record["context_name"]:20s} relevant premise: {p:20s}@{j:4d}')
            ncorrect_at_j += rel_at_j

            if ncorrect_at_j == len(all_pos_premise_names):
                K_at_full_recall = j + 1
                K_percent_at_full_recall = K_at_full_recall / len(retreived_premise_names)

            DCG += rel_at_j / discount
            p_at_j = ncorrect_at_j / (j + 1)
            AP += p_at_j * rel_at_j
        print(f'AP for {record["context_name"]:20s}: {AP:6f} | #pospremises: {all_pos_premise_names}')
        collator.APs.append(AP)
        NDCG = DCG / IDCG
        collator.NDCGs.append(NDCG)
        collator.Ks_at_full_recall.append(K_at_full_recall)
        collator.Ks_percent_at_full_recall.append(K_percent_at_full_recall)

        examples.append({
            "total_num_premises": len(ix2premise),
            "context_name": record["context_name"],
            "all_pos_premise_names": record["all_pos_premise_names"],
            "retreived_premise_names": [{ "num_entries": len(lvl), "entries": lvl[:10] } for lvl in level2names[:4]], # only keep 10 per level.
            "level2recall": level2recall,
            "level2cumulative_seen": level2cumulative_seen,
            "TP1" : TP1,
            "TP10": TP10,
            "R10" : R10,
            "R1" : R1,
            "NDCG": NDCG,
            "RR" : RR,
            "AP": AP
        })

    # finish processing all records.
    # average NDCG
    R1 = np.mean(collator.R1s)
    R10 = np.mean(collator.R10s)
    MAP = np.mean(collator.APs)
    NDCG = np.mean(collator.NDCGs)
    RR = np.mean(collator.RRs)
    logger.info(f"** Eval on {validate_dataset} | R1[0-1]: {R1} , RR[0-1]: {RR}, R10[0-1]: {R10}, MAP[0-1]: {MAP}, NDCG[0-1]: {NDCG} **")

    record_path = pathlib.Path(model_dir) / "sine_qua_non.record.json"
    with open(record_path, "w") as f:
        json.dump({ "examples": examples }, f, indent=1)



def me_po(test_model_path : str, 
        model_dir : str,
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    # paper reference: Lightweight relevance filtering for machine-generated resolution problems
    # ~ MEng and PaulsOn
    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{embeds_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]
    assert embeds is not None
    embedded_strings = list(embeds.keys())
    logger.info(f"load embeds. {len(embedded_strings)}")

    logger.info(f"starting model testing from path '{data_dir}'")
    assert embeds is not None

    validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0,
                                                     num_in_file_negatives=0,
                                                    embedded_strings=embedded_strings)

    # TODO: don't throw away stuff, just truncate.
    # version of model.test() that implements sine selection
    ix2premise_name = [name for name in corpus.all_premise_names \
                       if corpus.get_premise_embed_str_for_name(name) in embedded_strings]
    ix2premise = [corpus.get_premise_embed_str_for_name(name) for name in ix2premise_name]

    # what are the precise rules here? What do we choose to be the axioms? I guess the full premise DB
    mepo = MePo(symbols=ix2premise_name, axioms=dict(zip(ix2premise_name, ix2premise)))
    premise2ix = { ix2premise[ix] : ix for ix in range(len(ix2premise)) }

    seen_contexts = set()
    collator = TestStatisticsCollator()
    examples = []
    for record in tqdm(validate_dataset, desc=f"evaluation on {validate_dataset}"):
        if record["context_name"] in seen_contexts: continue # skip repeated contexts.
        # this is already sorted in descending order.
        rank2namescore = mepo.goal2selection(corpus.get_ctx_embed_str_for_name(record["context_name"]))
        rank2name = [name for (name, score) in rank2namescore]

        retreived_premise_names = rank2name
        all_pos_premise_names = record["all_pos_premise_names"]
        all_pos_premise_names_set = set(all_pos_premise_names)
        # TODO: filter by accessible premises.
        TP1 = retreived_premise_names[0] in all_pos_premise_names
        R1 = float(TP1) / 1
        collator.R1s.append(R1)
        TP10 = len(all_pos_premise_names_set.intersection(retreived_premise_names[:10]))
        R10 = float(TP10) / len(all_pos_premise_names[:10])
        collator.R10s.append(R10)

        RR = 0
        for j, p in enumerate(retreived_premise_names):
            if p in all_pos_premise_names:
                RR = (1.0 / (j + 1))
                break
        collator.RRs.append(RR)

        # AP = integral_0^1 P(r) dr
        # change of variables into k 
        # let rel(k) = 1 if retreived_premises[k] in all_pos_premises else 0
        # let s(k) = sum_{j=0}^k rel(j)
        # p(k) = s(k) / k
        # r = s(k) / |all_pos_premises|
        # dk = 1
        # dr = (r(k + dk) -  r(k)) / dk 
        #    = (r(k + 1) - r(k)) / 1
        #    = rel(k+1) / |all_pos_premises|
        # AP = integral_0^1 P(r) dr
        #    = sum_0^N P(r(k)) dr(k)
        #    = sum_0^N p(k) rel(k) / |all_pos_premises|
        AP = 0
        DCG = 0
        IDCG = 0
        
        K_at_full_recall = np.nan # How to correctly initialize?
        K_percent_at_full_recall = np.nan
        ncorrect_at_j = 0

        for j, p in enumerate(retreived_premise_names):
            discount = np.log2(j + 1) if j > 0 else 1
            if j < len(all_pos_premise_names):
                IDCG += 1.0 / discount      

            rel_at_j = int(p in all_pos_premise_names)
            if rel_at_j:
                print(f'ctx: {record["context_name"]:20s} relevant premise: {p:20s}@{j:4d}')
            ncorrect_at_j += rel_at_j

            if ncorrect_at_j == len(all_pos_premise_names):
                K_at_full_recall = j + 1
                K_percent_at_full_recall = K_at_full_recall / len(retreived_premise_names)

            DCG += rel_at_j / discount
            p_at_j = ncorrect_at_j / (j + 1)
            AP += p_at_j * rel_at_j
        print(f'AP for {record["context_name"]:20s}: {AP:6f} | #pospremises: {all_pos_premise_names}')
        collator.APs.append(AP)
        NDCG = DCG / IDCG
        collator.NDCGs.append(NDCG)
        collator.Ks_at_full_recall.append(K_at_full_recall)
        collator.Ks_percent_at_full_recall.append(K_percent_at_full_recall)

        examples.append({
            "total_num_premises": len(ix2premise),
            "context_name": record["context_name"],
            "all_pos_premise_names": record["all_pos_premise_names"],
            "retreived_premise_names": rank2namescore[:20],
            "TP1" : TP1,
            "TP10": TP10,
            "R10" : R10,
            "R1" : R1,
            "NDCG": NDCG,
            "RR" : RR,
            "AP": AP
        })

    # finish processing all records.
    # average NDCG
    R1 = np.mean(collator.R1s)
    R10 = np.mean(collator.R10s)
    MAP = np.mean(collator.APs)
    NDCG = np.mean(collator.NDCGs)
    RR = np.mean(collator.RRs)
    logger.info(f"** Eval on {validate_dataset} | R1[0-1]: {R1} , RR[0-1]: {RR}, R10[0-1]: {R10}, MAP[0-1]: {MAP}, NDCG[0-1]: {NDCG} **")

    record_path = pathlib.Path(model_dir) / "me_po.record.json"
    with open(record_path, "w") as f:
        json.dump({ "examples": examples }, f, indent=1)

def debug_missing_key(
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    corpus  = Corpus(corpus_path, import_graph_path)
     #logger.debug(f"loading embeds '{embeds_path}'...")


    strings = set()
    # check how many we skip
    for name in tqdm(corpus.all_names(), desc="querying openAI"):
        logger.info(f"processing '{name}'")
        ctx = corpus.get_ctx_embed_str_for_name(name)
        premise = corpus.get_premise_embed_str_for_name(name)
        strings.add(ctx)
        strings.add(premise)

    logger.debug("done querying. Now building gzip record")
    strings = sorted(list(strings)) # is this deterministic? probably not.
    for strings_batch in tqdm(list(openai_batch(strings)), desc="building openAI embeddings"):
        pass
    print(f"#strings skipped by OpenAI batching: {OPENAI_NSKIPPED:20d} | percentage: {OPENAI_NSKIPPED / len(strings) * 100.0:4.2f}")
    _ = input("press key to continue>")

    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]



    # seach for missing keys from corpus
    logger.info(f"===searching for missing names in corpus=====")
    tofind = []
    for name in tqdm(corpus.all_names(), desc="building names dict"):
        tofind.append(corpus.get_ctx_embed_str_for_name(name))
        tofind.append(corpus.get_premise_embed_str_for_name(name))

    missing_names = sorted(list(set([n for n in tofind if n not in embeds.keys()])))
    print(f"#names missing from corpus: {len(missing_names)}")
    for missing_name in tqdm(missing_names, desc="missing names search close match"):
        logger.info(f"===searching for close matches to {missing_name[:30]}===")
        for (ix, known_name) in enumerate(embeds.keys()):
            if missing_name[:256] in known_name:
                print(f"found matching name at ix: {ix:5d}")
                # diff = difflib.context_diff(known_name, missing_name)
                # print(''.join(diff), end="")


    logger.info(f"===searching for missing names in train dataset=====")
    train_dataset = PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "train.json"],
                                    corpus=corpus,
                                    is_train=True,
                                    num_negatives=2, num_in_file_negatives=1)
    tofind = []
    for record in tqdm(train_dataset):
        print(f"record: {record}")
        tofind.append(corpus.get_ctx_embed_str_for_name(record["context_name"]))
        tofind.append(corpus.get_premise_embed_str_for_name(record["pos_premise_name"]))
    missing_names = sorted(list(set([n for n in tofind if n not in embeds.keys()])))
    print(f"#names missing from dataset: {len(missing_names)}")
    for missing_name in tqdm(missing_names, desc="missing names search close match"):
        logger.info(f"===searching for close matches to name missing in dataset {missing_name[:30]}===")
        for (ix, known_name) in enumerate(embeds.keys()):
            if missing_name[:256] in known_name:
                print(f"found matching name at ix: {ix:5d}")
                # diff = difflib.context_diff(known_name, missing_name)
                # print(''.join(diff), end="")


def call_fn_with_dict(f : Callable, d : Dict[str, Any]):
    sig = inspect.signature(f)
    calldict = dict()
    for p in sig.parameters:
        if p not in d:
            logger.error(f"function '{f}' expects argument '{p}'. Not found in argument dict '{d}'")
            raise RuntimeError("unable to find function argument")
        calldict[p] = d[p]
    logger.info(f"invoking function {f} with args {calldict}")
    return f(**calldict)


def toplevel(args):
    opts_all = yaml.safe_load(args.config)
    logger.info(f"opts: {opts_all}")
    opts_common = opts_all["common"]
    if args.command == "download":
        opts_download = opts_all["download"]
        opts = {**opts_common, **opts_download}
        call_fn_with_dict(download, opts)
    elif args.command =="fit":
        opts_fit = opts_all["fit"]
        opts = {**opts_common, **opts_fit}
        call_fn_with_dict(fit, opts)
    elif args.command =="test":
        opts_test = opts_all["test"]
        opts = {**opts_common, **opts_test}
        call_fn_with_dict(test, opts)
    elif args.command =="sinequanon":
        opts = opts_common
        call_fn_with_dict(sine_qua_non, opts)
    elif args.command =="mepo":
        opts = opts_common
        call_fn_with_dict(me_po, opts)
    elif args.command =="debug_missing_key":
        call_fn_with_dict(debug_missing_key, opts_common)
    else:
        logger.error(f"ERROR: expected one of 'download', 'fit', 'test', 'sinequanon' found: '{args.command}'")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
                    prog='model')
    parser.add_argument('--config', type=argparse.FileType('r'), required=True)
    parser.set_defaults(command=None)

    # download vectors from openAI embeddings.
    subparsers = parser.add_subparsers()
    download = subparsers.add_parser('download')
    download.set_defaults(command="download")

    # run training loop of linear layer.
    fit = subparsers.add_parser('fit')
    fit.set_defaults(command="fit")

    # run cosine similarity.
    test = subparsers.add_parser('test')
    test.set_defaults(command="test")

    sine = subparsers.add_parser('sinequanon')
    sine.set_defaults(command="sinequanon")

    # run cosine similarity.
    debug_missing_key = subparsers.add_parser('debug_missing_key')
    debug_missing_key.set_defaults(command="debug_missing_key")

    args = parser.parse_args()
    toplevel(args)

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
