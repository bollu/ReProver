#!/usr/bin/env python3
import yaml
import argparse
import sys
from dataclasses import dataclass, asdict, field
from typing import *
import os
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
import pathlib
import retrievalfstar.datamodule

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# encoder = tiktoken.get_encoding(embedding_encoding)

# openai.organization = os.getenv("OPENAI_ORGANIZATION") #  "org-BkKFe6jDEL6tnBM8uzo3q0MC"
openai.api_key = os.getenv("OPENAI_EMBEDDINGS_API_KEY")
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
                logger.warning("skipping string at index '{ix}'")
                ix += 1 

            if ntokens + len(strings[ix]) >= max_tokens:
                # batch doesn't have enough space.
                break 

            logger.debug(f"building batch, ix={ix}, ntokens={ntokens}")
            ntokens += len(strings[ix]) #upper bound for actual number of tokens.
            batch.append(strings[ix])
            ix += 1

        logger.debug(f"yielding batch")
        yield batch

@dataclass
class Model:
    # indexed by string index.
    embeddings : torch.tensor = None
    minibatch_size : int
    dl : DataLoader
    num_out_of_file_neg_samples : int
    num_in_file_neg_samples : int

    def initialize_embeddings(self, num_embeddings: int, embedding_dim: int):
        self.embeddings = torch.Tensor([num_embeddings, embedding_dim])

    def add_word(self, ix: LocatedIx, embedding : torch.Tensor, dl : DataLoader):
        self.embeddings[ix] = embedding

    def add_words(self, ixs : List[LocatedIx], embeddings : List[torch.Tensor]):
        for (ix, embed) in zip(ixs, embeddings):
            self.embeddings[ix] = embed

    def train_microbatch(self, data: Dict[str, Any]) -> torch.Tensor:
        loss = 0
        for record in records:
            pass
        return loss

    def make_microbatch(self, records: List[Record]) -> Dict[str, Any]:
        out = dict()
        out["context"] = []
        out["premise"] = []
        out["similarity"] = []
        # TODO: create negative samples.
        for r in records:
            for p in r.pos_premise_ix:
                out["context"].append(self.embeddings[r.context_ix])
                out["premise"].append(self.embeddings[p])
                out["similarity"].append(1.0)

            nsamples = 0
            while nsamples < self.num_in_file_neg_samples:
                out["context"].append(self.embeddings[r.context_ix])
                p = self.dl.corpus.get_in_file_negative_sample(r.corpus_ix)
                if p in r.all_pos_premise_ixs: continue
                nsamples += 1
                out["premise"].append(self.embeddings[p])
                out["similarity"].append(0.0)


            nsamples = 0
            while nsamples < self.num_out_of_file_neg_samples:
                out["context"].append(self.embeddings[r.context_ix])
                p = self.dl.corpus.get_out_of_file_negative_sample(r.corpus_ix)
                if p in r.all_pos_premise_ixs: continue
                nsamples += 1
                out["premise"].append(self.embeddings[p])
                out["similarity"].append(0.0)
        out["context"] = torch.cat(out["context"])
        out["premise"] = torch.cat(out["premise"])



    def train_minibatch(self, records: List[Record]) -> float:
        loss = 0
        for i in range(len(records), self.microbatch_size):
            loss += self.train_microbatch(records[i:i+self.microbatch_size]))
        loss.backward()
        logger.debug(f"batch loss: {loss.item()}")
        return loss.item()

    def train_epoch_ (self, records: List[Record]) -> float
        epoch_loss = 0
        for i in range(len(records), self.minibatch_size):
            epoch_loss += self.train_minibatch(records[i:i+self.minibatch_size])
        logger.debug(f"epoch loss: {loss.item()}")
        return epoch_loss

    def train(self):
        for e in range(nepochs):
            epoch_loss = self.train_epoch_(self.dl.records)


    def test(self):
    for data_path in data_paths:
        dl = DataLoader()
        dl.load_data_from_file(path)

        logger.info(f"running evaluation on {path}")
        R1 = []
        R10 = []
        MAP = []
        NDCG = []
        IDCG = []
        collator = TestStatisticsCollator()

        for record in tqdm(dl.records, desc=f"evaluation on {path}"):
            similarities = model.embeddings[record.context_i] @ all_premises_embeds
            idxs = similarities.argsort(dim=1, descending=True).tolist()

            TP1 = retrieved_premises[0] in all_pos_premises
            R1 = float(TP1) / len(all_pos_premises)
            collator.R1s.append(R1)
            TP10 = len(all_pos_premises_set.intersection(retrieved_premises[:10]))
            R10 = float(TP10) / len(all_pos_premises)
            collator.R10s.append(R10)

            RR = 0
            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    RR = (1.0 / (j + 1))
                    break
            collator.RRs.append(RR)

            # AP = integral_0^1 P(r) dr
            # change of variables into k 
            # let rel(k) = 1 if retrieved_premises[k] in all_pos_premises else 0
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

            for j, p in enumerate(retrieved_premises):
                discount = np.log2(j + 1) if j > 0 else 1
                if j < len(all_pos_premises):
                    IDCG += 1.0 / discount          

                rel_at_j = int(p in all_pos_premises)
                ncorrect_at_j += rel_at_j

                if ncorrect_at_j == len(all_pos_premises):
                    K_at_full_recall = j + 1
                    K_percent_at_full_recall = K_at_full_recall / len(retrieved_premises)
                DCG += rel_at_j / discount
                p_at_j = ncorrect_at_j / (j + 1)
                AP += p_at_j * rel_at_j
            AP /= len(all_pos_premises)
            collator.APs.append(AP)
            NDCG = DCG / IDCG
            collator.NDCGs.append(NDCG)
            collator.Ks_at_full_recall.append(K_at_full_recall)
            collator.Ks_percent_at_full_recall.append(K_percent_at_full_recall)

        # finish processing all records.
        # average NDCG
        R1 = np.mean(collator.R1)
        R10 = np.mean(collator.R10)
        MAP = np.mean(collator.MAP)
        NDCG = np.mean(collator.NDCG)
        logger.info(f"** Eval on {path} | R1: {R1} % , R10: {R10} %, MAP: {MAP} %, NDCG: {NDCG} % **")

class OutputDirectory:
    basedir_path : pathlib.Path = pathlib.Path("pretrained_logs")
    version_dir_name : str = "version_"
    curdir : pathlib.Path

    def __init__(self):
        os.makedirs(self.basedir_path, exist_ok=True)
        dirnames = os.listdir(path=self.basedir_path)

        max_ix = max([int(d.split(self.version_dir_name)[1]) for d in dirnames], default=1)
        self.curdir = self.basedir_path / f"{self.version_dir_name}{max_ix+1}"
        os.makedirs(self.curdir)

    def get_curdir_path(self) -> pathlib.Path:
        """get basepath for saving data"""
        return self.curdir

    def copy_file(self, filepath : str):
        """copy a file into the output directory"""
        filepath = pathlib.Path(filepath)
        with open(self.curdir / filepath.name, "w") as fout:
            fout.write(open(filepath).read())

def download(output_dir : OutputDirectory, data_path : str):
    dl = DataLoader()
    dl.load_data_from_files([pathlib.Path(data_path) / "train.json",
                             pathlib.Path(data_path) / "test.json",
                             pathlib.Path(data_path) / "validate.json"])
    model = Model()
    strings = set() # is this deterministic when converting to list?

    for record in tqdm(dl.records, desc="querying openAI"):
        ctx = dl.corpus.ix2def(record.context_ix)
        premise = dl.corpus.ix2def(record.pos_premise_ix)
        # can have empty definition from premises whose definitions we don't know.
        if ctx.definition:
            strings.add(ctx.definition)
        strings.add(premise.name)

    logger.debug("done querying. Now building gzip record")
    strings = list(strings) # is this deterministic? probably not.
    strings.sort()
    for strings_batch in tqdm(list(openai_batch(strings)), desc="building openAI embeddings"):
        assert len(strings_batch)
        logger.debug(f"vvbatch size: {len(strings_batch)}vv")
        logger.debug(f"\n".join([f"  - '{s}'" for s in strings_batch]))
        logger.debug(f"--")
        vecs = get_embeddings(strings_batch, engine=embedding_model)
        model.add_words(strings_batch, vecs)


    logger.debug("dumping model.")
    with gzip.open(output_dir.get_curdir_path() / "embeddings.pickle.gz", "wb") as f:
        pickle.dump({ "model": model }, f)
    logger.debug("dumped!")

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

def test(output_dir : OutputDirectory, ckpt_path : str, data_path : str, eval_batch_size : int):
    ckpt_path = pathlib.Path(ckpt_path)
    logger.debug("loading pickle...")
    with gzip.open(ckpt_path, "rb") as f:
        loaded = pickle.load(f)
        model = loaded["model"]
    assert model is not None
    assert all_dl is not None
    assert isinstance(model, Model)
    assert isinstance(all_dl, DataLoader)
    print(f"loaded pickle! model #strings: '{len(model.strings)}'")





def fit(output_dir : OutputDirectory, ckpt_path : str, data_path : str, train_batch_size : int, minibatch_size : int, nepoch : int):
    ckpt_path = pathlib.Path(ckpt_path)
    logger.debug("loading pickle...")
    with gzip.open(ckpt_path, "rb") as f:
        loaded = pickle.load(f)
        model = loaded["model"]
    assert model is not None
    assert dl is not None
    model.train()
    



def toplevel():
    output_dir = OutputDirectory()
    output_dir.copy_file(os.path.realpath(args.config.name))

    opts_all = yaml.safe_load(args.config)["data"]
    opts_common = opts_all["common"]
    if args.command == "download":
        opts_download = opts_all["download"]
        opts = {**opts_common, **opts_download}
        download(output_dir=output_dir, **opts)

    if args.command =="fit":
        opts_fit = opts_all["fit"]
        opts = {**opts_fit, **opts_download}
        fit(output_dir=output_dir, **opts)
    
    if args.command =="test":
        opts_test = opts_all["test"]
        opts = {**opts_test, **opts_download}
        test(output_dir=output_dir, **opts)
        return

    print(f"ERROR: expected one of 'download' or 'test' commands, found: '{args.command}'")
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
