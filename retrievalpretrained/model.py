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
class Loc:
    line : int
    col : int 

    @staticmethod
    def unknown():
        return Loc(line=0, col=0)

@dataclass
class Range:
    start_loc : Loc 
    end_loc : Loc


    @staticmethod
    def unknown():
        return Range(start_loc=Loc.unknown(),
                     end_loc=Loc.unknown())

UNKNOWN_FILE_PATH="<UNK>"
@dataclass
class Located:
    file_path : str
    file_range: Range
    name : str # the name of the object that is located.


@dataclass
class Definition(Located):
    type_ : str
    definition : str

    def to_str(self):
        return 

LocatedIx = int
@dataclass
class Corpus:
    import_graph: nx.DiGraph
    definitions: List[Located]
    name2def : Dict[str, LocatedIx] # ix of located in definitions.

    def __init__(self):
        self.definitions = []
        self.name2def = dict()

    def add_definition(self, d : Located) -> LocatedIx: 
        if d.name in self.name2def: 
            raise AssertionError(f"double definition of definition '{d.name}'")
        ix = len(self.definitions)
        self.definitions.append(d)
        self.name2def[d.name] = ix
        return ix


    def try_def2ix (self, name : str) -> Optional[int]:
        if name in self.name2def:
            return self.name2def[name]
        return None

    def def2ix (self, name : str) -> Optional[int]:
        if name in self.name2def:
            return self.name2def[name]
        raise RuntimeError(f"expected to find definition '{name}', was unable to find")

    def ix2def(self, ix : LocatedIx) -> Located:
        assert ix < len(self.definitions)
        return self.definitions[ix]

@dataclass
class Record:
    context_ix : LocatedIx
    pos_premise_ix : LocatedIx
    all_pos_premise_ixs : List[LocatedIx]

class DataLoader:
    corpus : Corpus
    records : List[Record]
    all_premise_ixs : List[LocatedIx] # all premises ever

    def __init__(self):
        self.corpus = Corpus()
        self.records = []

    @staticmethod
    def load_json_or_jsonl(path : str) -> List[Dict[str, Any]]:
        ext = pathlib.Path(path).suffix
        if ".json" == ext:
            return json.load(open(path))
        else:
            assert ".jsonl" == ext
            return [json.loads(line) for line in open(path)]

    def load_data_from_file(self, data_path: str):
        logger.info(f"Loading data from '{data_path}'")
        in_data = DataLoader.load_json_or_jsonl(data_path)
        # First pass: add every definition to the corpus
        for (i, datum) in tqdm(enumerate(in_data), desc="building corpus"):
            if i == 0:
                logger.debug(f"loading datum with keys '{datum.keys()}'")
                logger.debug(f"loading '{i}'th data from '{data_path}'. keys: '{datum.keys()}'")

            if datum["name"] in self.corpus.name2def:
                logger.warning(f'already have definition {datum["name"]}. skipping.')
                continue

            d = Definition(
                file_path = datum["file_name"],
                file_range = Range(
                    start_loc=Loc(
                        line = datum["start_line"],
                        col=datum["start_col"]),
                    end_loc=Loc(
                        line=datum["end_line"],
                        col=datum["end_col"])),

                name = datum["name"],
                type_ = datum["type"],
                definition = datum["definition"])
            self.corpus.add_definition(d)

        # Second pass: generate training records.
        ntotal = 0
        nfaked = 0
        for (i, datum) in tqdm(enumerate(in_data), desc="building training records"):
            skip = False
            ntotal += 1
            context_ix = self.corpus.try_def2ix(datum["name"])
            if context_ix is None:
                logger.error(f'expected to find definition {datum["name"]}, unable to find datum')
                raise AssertionError('unable to find definition {datum["name"]}')

            all_pos_premise_ixs = []
            for p in datum["premises"]:
                ix = self.corpus.try_def2ix(p)
                if ix is None:
                    logger.warning(f'in datum {datum["name"]}, unable to find premise {p}. Creating fake definition')
                    ix = self.corpus.add_definition(
                        Definition(file_path=UNKNOWN_FILE_PATH,
                                  file_range=Range.unknown(),
                                  name=p,
                                  type_="",
                                  definition=""))
                    nfaked += 1
                assert ix is not None
                all_pos_premise_ixs.append(ix)
                self.all_premise_ixs.append(ix)

            if skip: continue
            for pos_premise_ix in all_pos_premise_ixs:
                self.records.append(Record(
                    context_ix=context_ix,
                    pos_premise_ix=pos_premise_ix,
                    all_pos_premise_ixs=all_pos_premise_ixs
                ))       
        # end of loop over data
        logger.debug(f"faked {nfaked}/{ntotal} = {nfaked/ntotal*100:4.2f} %")

    def load_data_from_files(self, data_paths: List[str]):
        for path in data_paths:
            self.load_data_from_file(path)

                    
@dataclass
class Model:
    embeddings : List[torch.Tensor] = field(default_factory=list)
    strings : List[str] = field(default_factory=list)


    def add_word(self, s : str, embedding : torch.Tensor):
        self.strings.append(s)
        self.embedding.append(embedding)

    def add_words(self, ss : List[str], embeddings : List[torch.Tensor]):
        self.strings.extend(ss)
        self.embeddings.extend(embeddings)


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

def download(output_dir : OutputDirectory, data_path : str, batch_size : int, eval_batch_size : int):
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
    for strings_batch in tqdm(list(openai_batch(strings)), desc="building openAI embeddings"):
        assert len(strings_batch)
        logger.debug(f"vvbatch size: {len(strings_batch)}vv")
        logger.debug(f"\n".join([f"  - '{s}'" for s in strings_batch]))
        logger.debug(f"--")
        vecs = get_embeddings(strings_batch, engine=embedding_model)
        model.add_words(strings_batch, vecs)


    logger.debug("dumping model.")
    with gzip.open(output_dir.get_curdir_path() / "embeddings.pickle.gz", "wb") as f:
        # TODO: consider also dumping the dataloader since it contains the ordering of the data?
        pickle.dump({"model": model, "dl" : dl}, f)
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
def test(output_dir : OutputDirectory, ckpt_path : str, data_path : str, batch_size : int, eval_batch_size : int):
    ckpt_path = pathlib.Path(ckpt_path)
    logger.debug("loading pickle...")
    with gzip.open(ckpt_path, "rb") as f:
        loaded = pickle.load(f)
        model = loaded["model"]
        all_dl = loaded["dl"]
    assert model is not None
    assert all_dl is not None
    assert isinstance(model, Model)
    assert isinstance(all_dl, DataLoader)
    print(f"loaded pickle! model #strings: '{len(model.strings)}'")

    data_paths = [pathlib.Path(data_path) / "train.json",
             pathlib.Path(data_path) / "test.json",
             pathlib.Path(data_path) / "validate.json"])

    all_dl = DataLoader()
    all_dl.load_data_from_files(data_paths) # load all data for all premises.
    all_premises_embeds = torch.tensor.cat([model.embeddings[pix] for pix in all_dl.all_premise_ixs])
    for data_path in data_paths:
        dl = DataLoader()
        dl.load_data_from_file(path)

        logger.info(f"running evaluation on {path}")
        R1 = []
        R10 []
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

def toplevel(args: argparse.Namespace):
    output_dir = OutputDirectory()
    output_dir.copy_file(os.path.realpath(args.config.name))

    opts = yaml.safe_load(args.config)["data"]
    if args.command == "download":
        download(output_dir=output_dir, **opts)
        return
    
    if args.command =="test":
        test(output_dir=output_dir, ckpt_path=args.ckpt_path, **opts)
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
    test.add_argument("--ckpt_path", required=True, type=str)
    test.set_defaults(command="test")

    args = parser.parse_args()
    toplevel(args)

if __name__ == "__main__":
    main()
