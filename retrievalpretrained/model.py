#!/usr/bin/env python3
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

def openai_batch(strings :  List[str], max_tokens : int = 3000, batch_size : int = 16):
    """
    batch the list of strings according to openAI batching rules,
    which is that each batch can have at most 5000 tokens,
    and a batch size of at most 2000. Returns the indexes
    of the strings that were skipped as they are too long.
    TODO: consider giving this information eagerly.
    """
    # actual max tokens is 8192
    skipped = []
    ix = 0
    while ix < len(strings):
        batch = []
        ntokens = 0
        while ix < len(strings) and len(batch) < batch_size:

            if len(strings[ix]) >= max_tokens:
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
    num_negatives : int
    is_train : bool
    def __init__(
        self,
        data_paths: List[str],
        corpus : Corpus,
        is_train: bool,
        num_negatives : int,
        num_in_file_negatives : int,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.is_train = is_train
        assert len(data_paths) == 1
        data_path = data_paths[0]
        self.data_path = data_path
        self.data = PretrainedDataset._load_data(corpus=self.corpus, data_path=data_paths[0])
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
    def _load_data(corpus : Corpus, data_path: str) -> List[Dict[str, Any]]:
        out_data = []
        logger.info(f"Loading data from '{data_path}'")
        in_data = PretrainedDataset.load_json_or_jsonl(data_path)
        for (i, datum) in tqdm(enumerate(in_data)):
            if i == 0:
                logger.info(f"loading datum with keys '{datum.keys()}'")
            logger.info(f"loading '{i}'th data from '{data_path}'. keys: '{datum.keys()}'")
            if not corpus.has_definition_for_name(datum["name"]):
                logger.warning(f"skipping defn '{datum['name']}' as we do not have definition")
                continue
            # TODO: refactor this to be way flatter.
            # context = Context(name=datum["name"],
            #     type_=datum["type"],
            #     definition=datum["definition"])
            # all premises that are *used* in the tactic.
            all_pos_premise_names = datum["premises"]
            # only keep those premise names that don't occur in the ctx embedding.
            all_pos_premise_names = \
                    [p for p in datum["premises"] if p not in corpus.get_ctx_embed_str_for_name(datum["name"])]
            if not all_pos_premise_names:
                continue # skip this def cause it has zero premises
            know_all_premise_defs = all([corpus.has_definition_for_name(p) for p in all_pos_premise_names])
            if not know_all_premise_defs:
                continue # skip this def cause its premise does not occur
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

            ex["neg_premise_names"] = random.sample(premises_outside_file, self.num_negatives)
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
    ctx_projection : nn.Module
    premise_projection : nn.Module
    on_epoch_end_callbacks : List[Callable] # callback args: (model=MyModel, epoch_loss=epoch_los)

    def __init__(self,
                 corpus : Corpus,
                 embeddings : Dict[str, torch.tensor], 
                 train_dataset : PretrainedDataset,
                 validate_dataset : PretrainedDataset,
                 nepoch : int,
                 minibatch_size : int,
                 microbatch_size : int): 
        self.corpus = corpus
        self.embeddings = embeddings
        self.nepoch = nepoch
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.minibatch_size = minibatch_size
        self.microbatch_size = microbatch_size
        self.ctx_projection = MLP(OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN)
        self.premise_projection = MLP(OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN, OPENAI_EMBEDDING_VEC_LEN)
        self.on_epoch_end_callbacks = []

    def mk_pickle_dict(self):
        return { "ctx_projection" : self.ctx_projection.state_dict(), "premise_projection" : self.premise_projection.state_dict() }

    def load_pickle_dict(self, d):
        self.ctx_projection.load_state_dict(d["ctx_projection"])
        self.premise_projection.load_state_dict(d["premise_projection"])

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
            ctxs.append(self.embeddings[self.corpus.get_ctx_embed_str_for_name(c)].reshape(1, -1))
        for (ip, p) in enumerate(premise_names):
            premises.append(self.embeddings[self.corpus.get_premise_embed_str_for_name(p)].reshape(1, -1))

        for (ic, c) in enumerate(ctx_names):
            labels.append([])
            for (ip, p) in enumerate(premise_names):
                labels[ic].append(float(p in data[ic]["all_pos_premise_names"]))
            labels[ic] = torch.tensor(labels[ic]).reshape(1, -1)

        ctxs = torch.cat(ctxs, axis=0)
        premises = torch.cat(premises, axis=0)
        labels = torch.cat(labels, axis=0)

        logger.info(f"ctxs: {ctxs.shape} | premises: {premises.shape} | labels: {labels.shape}")
        # | ctx_embed: {ctx_embed.shape} | premise_embed: {premise_embed.shape} | dots: {dots.shape}")

        # [NCTX x DEMBED]
        ctx_embed = self.ctx_projection(ctxs)
        # [NPREMSE x DEMBED]
        premise_embed = self.premise_projection(premises)
        logger.info(f"ctxs_embed {ctx_embed.shape} | premise_embed: {premise_embed.shape}")
        # [NCTX x NPREMISE]
        dots = ctx_embed @ premise_embed.T
        logger.info(f"dots: {dots.shape} | labels: {labels.shape}")
        loss = torch.nn.functional.mse_loss(dots, labels)
        return loss

    def train_minibatch(self, records: List[Dict[str, Any]]) -> float:
        loss = 0
        for i in range(0, len(records), self.microbatch_size):
            loss += self.train_microbatch(records[i:i+self.microbatch_size])
        loss.backward()
        loss_val = loss.item()
        del loss
        return loss_val

    def train_epoch(self, iepoch: int, records: List[Dict[str, Any]]) -> float:
        epoch_loss = 0
        for i in range(0, len(records), self.minibatch_size):
            logger.info(f".. ..minibatch[{i}:{i+self.minibatch_size}]")
            batch_loss = self.train_minibatch(records[i:i+self.minibatch_size])
            epoch_loss += batch_loss
            logger.info(f"epoch[0-1] {iepoch/self.nepoch:4.2f}, batch[0-1] {i/len(records):4.2f}, batch loss: {batch_loss:5.3f}, epoch_loss: {epoch_loss:5.3f}")
        # logger.info(f"epoch loss: {loss.item()}")
        return epoch_loss

    def train(self):
        for e in range(self.nepoch):
            logger.info(f"..epoch: {e}/{self.nepoch}")
            epoch_loss = self.train_epoch(e, self.train_dataset)
            logger.info(f"..epoch end: {e}/{self.nepoch}, loss: {epoch_loss}")
            for callback in self.on_epoch_end_callbacks:
                callback(model=self, epoch=e, epoch_loss=epoch_loss)


    def test(self):
        logger.info(f"running evaluation on {self.validate_dataset}")
        collator = TestStatisticsCollator()

        ix2premise_name = list(self.corpus.all_premise_names)
        ix2premise = [self.corpus.get_premise_embed_str_for_name(name) for name in ix2premise_name]
        premise2ix = { ix2premise[ix] : ix for ix in range(len(ix2premise)) }
        # NBATCH x OPENAI_EMBED_DIM
        all_premises_embeds = torch.cat([self.embeddings[ix2premise[ix]].view(1, -1) for ix in range(len(ix2premise)) ], axis=0)
        # NBATCH x OUR_EMBED_DIM
        all_premises_embeds = self.premise_projection(all_premises_embeds)
        logger.info("all_premises_embeds: {all_premises_embeds}")
        # premises indexed by ix.

        for record in tqdm(self.validate_dataset, desc=f"evaluation on {self.validate_dataset}"):
            # 1 x OPENAI_EMBED_DIM
            ctx_embed = self.embeddings[self.corpus.get_ctx_embed_str_for_name(record["context_name"])]
            # 1 x OUR_EMBED_DIM
            ctx_embed = self.ctx_projection(ctx_embed)
            # OUR_EMBED_DIM
            ctx_embed = ctx_embed.view(-1)
            # OUR_EMBED_DIM x [NBATCH x OUR_EMBED_DIM].T
            # OUR_EMBED_DIM x [OUR_EMBED_DIM; NBATCH]
            # NBATCH
            similarities = self.embeddings[self.corpus.get_ctx_embed_str_for_name(record["context_name"])] @ all_premises_embeds.T
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

        # finish processing all records.
        # average NDCG
        R1 = np.mean(collator.R1s)
        R10 = np.mean(collator.R10s)
        MAP = np.mean(collator.APs)
        NDCG = np.mean(collator.NDCGs)
        logger.info(f"** Eval on {self.validate_dataset} | R1[0-1]: {R1} , R10[0-1]: {R10}, MAP[0-1]: {MAP}, NDCG[0-1]: {NDCG} **")

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
    strings = list(strings) # is this deterministic? probably not.
    strings.sort()
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
    model_path : pathlib.Path
    lowest_loss : float
    def __init__(self, model_path : str):
        self.model_path = pathlib.Path(model_path)
        self.lowest_loss = np.inf

    def __call__(self, epoch : int, model: MyModel, epoch_loss : float):
        is_best_epoch =  epoch_loss < self.lowest_loss
        if is_best_epoch: 
            self.lowest_loss = epoch_loss
            save_path = self.model_path.parent / (self.model_path.stem + ".best" + self.model_path.suffix)
            logger.info(f"***saving best model to {save_path}***")
            with gzip.open(save_path, "wb") as f:
                pickle.dump(model.mk_pickle_dict(), f)

        logger.info(f"***saving model @ epoch {epoch} to {save_path}***")
        save_path = self.model_path.parent / (self.model_path.stem + f".epoch{epoch}" + self.model_path.suffix)
        with gzip.open(save_path, "wb") as f:
            pickle.dump(model.mk_pickle_dict(), f)



def fit(model_path : str, 
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{model_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]
    logger.info(f"load embeds. {len(list(embeds.keys()))}")

    logger.info(f"starting model training from path '{data_dir}'")
    assert embeds is not None
    model = MyModel(corpus=corpus,
                  embeddings=embeds,
                  minibatch_size=minibatch_size,
                  microbatch_size=microbatch_size,
                  nepoch=nepoch,
                  train_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "train.json"],
                                                  corpus=corpus,
                                                  is_train=True,
                                                  num_negatives=2, num_in_file_negatives=1),
                  validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0, num_in_file_negatives=0))
    logger.info(f"training model...")
    model.on_epoch_end_callbacks.append(ModelSaver(model_path))
    model.train()
    os.makedirs(pathlib.Path(model_path).parent, exist_ok=True)

    logger.info(f"writing model to {model_path}")
    with gzip.open(model_path, "wb") as f:
        pickle.dump(model.mk_pickle_dict(), f)


def test(model_path : str, 
        corpus_path : str, 
        import_graph_path : str,
        embeds_path : str,
        data_dir : str,
        minibatch_size : int, 
        microbatch_size : int,
        nepoch : int):
    corpus  = Corpus(corpus_path, import_graph_path)
    logger.debug(f"loading embeds '{model_path}'...")
    with gzip.open(embeds_path, "rb") as f:
        loaded = pickle.load(f)
        embeds = loaded["embeds"]
    logger.info(f"load embeds. {len(list(embeds.keys()))}")

    logger.info(f"starting model training from path '{data_dir}'")
    assert embeds is not None
    model = MyModel(corpus=corpus,
                  embeddings=embeds,
                  minibatch_size=minibatch_size,
                  microbatch_size=microbatch_size,
                  nepoch=nepoch,
                  train_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "train.json"],
                                                  corpus=corpus,
                                                  is_train=True,
                                                  num_negatives=2, num_in_file_negatives=1),
                  validate_dataset=PretrainedDataset(data_paths=[pathlib.Path(data_dir) / "validate.json"],
                                                     corpus=corpus,
                                                     is_train=False,
                                                     num_negatives=0, num_in_file_negatives=0))
    logger.info(f"loading model from {model_path}...")
    with gzip.open(model_path, "rb") as f:
        model.load_pickle_dict(pickle.load(f))
    logger.info(f"loaded model.")
    model.test()


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
    else:
        logger.error(f"ERROR: expected one of 'download' or 'test' commands, found: '{args.command}'")
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

    args = parser.parse_args()
    toplevel(args)

if __name__ == "__main__":
    main()
