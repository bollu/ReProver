"""Datamodule for the premise retrieval."""
import os
import json
import torch
import random
import itertools
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from lean_dojo import LeanGitRepo
from typing import Optional, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from lean_dojo.constants import LEAN3_DEPS_DIR, LEAN4_DEPS_DIR
from typing import *
import pprint

pp = pprint.PrettyPrinter(indent=2)

from commonfstar import Context, Corpus, Batch, Example, format_state, get_all_pos_premises

# {
#   "file_name": "Pulse.Checker.Pure.fst",
#   "start_line": 241,
#   "start_col": 46,
#   "end_line": 254,
#   "end_col": 65,
#   "definition": "fun g e t -> let fg = Pulse.Typing.elab_env g in let re = Pulse.Elaborate.Pure.elab_term e in let rt = Pulse.Elaborate.Pure.elab_term t in let _ = Pulse.Checker.Pure.catch_all #(FStar.Tactics.Types.typing_token fg re (FStar.Pervasives.Native.Mktuple2 #FStar.Tactics.Types.tot_or_ghost #FStar.Reflection.Types.typ FStar.Tactics.Types.E_Total rt)) (fun _ -> Pulse.Checker.Pure.rtb_core_check_term_at_type (Pulse.Typing.Env.push_context g \"core_check_term_with_expected_type\" (FStar.Reflection.V2.Builtins.range_of_term rt)) fg re rt) in (let FStar.Pervasives.Native.Mktuple2 #_ #_ topt issues = _ in FStar.Tactics.V2.Builtins.log_issues issues; (match topt with | FStar.Pervasives.Native.None #_ -> let _ = Pulse.Checker.Pure.ill_typed_term e (FStar.Pervasives.Native.Some #Pulse.Syntax.Base.term t) (FStar.Pervasives.Native.None #Pulse.Syntax.Base.term) in Pulse.Typing.Env.fail #(Pulse.Typing.typing g e t) g (FStar.Pervasives.Native.Some #Pulse.Syntax.Base.range (Mkterm?.range e)) _ | FStar.Pervasives.Native.Some #_ tok -> FStar.Reflection.Typing.T_Token (Pulse.Typing.elab_env g) (Pulse.Elaborate.Pure.elab_term e) (FStar.Pervasives.Native.Mktuple2 #FStar.Tactics.Types.tot_or_ghost #FStar.Reflection.Types.typ FStar.Tactics.Types.E_Total (Pulse.Elaborate.Pure.elab_term t)) (FStar.Squash.return_squash #(FStar.Tactics.Types.typing_token (Pulse.Typing.elab_env g) (Pulse.Elaborate.Pure.elab_term e) (FStar.Pervasives.Native.Mktuple2 #FStar.Tactics.Types.tot_or_ghost #FStar.Reflection.Types.typ FStar.Tactics.Types.E_Total (Pulse.Elaborate.Pure.elab_term t))) tok)) <: Pulse.Typing.typing g e t) <: Pulse.Typing.typing g e t",
#   "effect": "FStar.Tactics.Effect.Tac",
#   "effect_flags": [],
#   "hints": [
#     {
#       "hint_name": "Pulse.Checker.Pure.core_check_term_with_expected_type",
#       "hint_index": 1,
#       "fuel": 2,
#       "ifuel": 1,
#       "unsat_core": [
#         "@MaxIFuel_assumption",
#         "@query",
#         "FStar.Tactics.Types_pretyping_05a3bdeb4a1637ac1bf12ee84facf747",
#         "data_elim_FStar.Tactics.Result.Success",
#         "data_typing_intro_FStar.Pervasives.Native.Mktuple2@tok",
#         "data_typing_intro_FStar.Tactics.Types.E_Ghost@tok",
#         "equality_tok_FStar.Tactics.Types.E_Total@tok",
#         "equation_FStar.Reflection.Types.typ",
#         "equation_FStar.Tactics.Types.issues",
#         "equation_Pulse.Syntax.Base.range_singleton_trigger",
#         "fuel_guarded_inversion_FStar.Pervasives.Native.tuple2",
#         "fuel_guarded_inversion_FStar.Tactics.Result.__result",
#         "fuel_guarded_inversion_Pulse.Syntax.Base.term",
#         "function_token_typing_FStar.Reflection.Types.term",
#         "function_token_typing_FStar.Tactics.Types.issues",
#         "kinding_FStar.Pervasives.Native.option@tok",
#         "kinding_FStar.Tactics.Types.tot_or_ghost@tok",
#         "lemma_FStar.Pervasives.invertOption",
#         "typing_FStar.Pervasives.Native.__proj__Mktuple2__item___1",
#         "typing_FStar.Tactics.Types.typing_token",
#         "typing_Pulse.Elaborate.Pure.elab_term",
#         "typing_Pulse.Typing.elab_env",
#         "typing_tok_FStar.Tactics.Types.E_Total@tok"
#       ],
#       "query_elapsed_time": 0
#     }
#   ],
#   "mutual_with": [],
#   "name": "Pulse.Checker.Pure.core_check_term_with_expected_type",
#   "premises": [
#     "Pulse.Typing.Env.env",
#     "Pulse.Syntax.Base.term",
#     "FStar.Pervasives.Native.option",
#     "FStar.Tactics.Types.typing_token",
#     "FStar.Pervasives.Native.Mktuple2",
#     "FStar.Tactics.Types.tot_or_ghost",
#     "FStar.Reflection.Types.typ",
#     "FStar.Tactics.Types.E_Total",
#     "FStar.Tactics.Types.issues",
#     "Pulse.Typing.Env.fail",
#     "Pulse.Typing.typing",
#     "FStar.Pervasives.Native.Some",
#     "Pulse.Syntax.Base.range",
#     "Pulse.Syntax.Base.__proj__Mkterm__item__range",
#     "Prims.string",
#     "Pulse.Checker.Pure.ill_typed_term",
#     "FStar.Pervasives.Native.None",
#     "FStar.Reflection.Typing.T_Token",
#     "Pulse.Typing.elab_env",
#     "Pulse.Elaborate.Pure.elab_term",
#     "FStar.Squash.return_squash",
#     "Prims.unit",
#     "FStar.Tactics.V2.Builtins.log_issues",
#     "FStar.Pervasives.Native.tuple2",
#     "Pulse.Checker.Pure.catch_all",
#     "Pulse.Checker.Pure.rtb_core_check_term_at_type",
#     "Pulse.Typing.Env.push_context",
#     "FStar.Reflection.V2.Builtins.range_of_term",
#     "FStar.Reflection.Types.term",
#     "FStar.Reflection.Types.env"
#   ],
#   "proof_features": [],
#   "type": "g: Pulse.Typing.Env.env -> e: Pulse.Syntax.Base.term -> t: Pulse.Syntax.Base.term -> FStar.Tactics.Effect.Tac (Pulse.Typing.typing g e t)",
#   "is_lemma": false
# }

class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        num_negatives: int,
        num_in_file_negatives: int,
        max_seq_len: int,
        tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = list(
            itertools.chain.from_iterable(
                self._load_data(path) for path in data_paths
            )
        )

        # TODO: this should live in Corpus?
        self.all_premises = set()
        
        for datum in tqdm(self.data):
            self.all_premises.add(datum["pos_premise"])

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        logger.info(f"Loading data from {data_path}")

        # data_path should be a `.jsonl` file
        for (i, line) in tqdm(enumerate(open(data_path))):
            datum = json.loads(line)
            if i == 0:
                print(f"loading datum with keys '{datum.keys()}'")
            context = Context(name=datum["name"],
                type_=datum["type"],
                definition=datum["definition"])
            # all premises that are *used* in the tactic.
            all_pos_premises = datum["premises"]
            for pos_premise in all_pos_premises:
                data.append({
                    "context": context,
                    "pos_premise": pos_premise, # TODO: create an actual Premise object? smh.
                    "all_pos_premises": all_pos_premises, # all of the premises that this premise co-occurs with in this context.
                })       
                    
        # for line in tqdm(open(data_path, "r")):
        #     record = json.loads(record) 
        #     context = Context()
        #     all_pos_premises = get_all_pos_premises()
        # for thm in tqdm(json.load(open(data_path))):
        #     if thm["file_path"] in self.corpus:
        #         file_path = thm["file_path"]
        #     else:
        #         # The theorem is from a dependency.
        #         _, repo_name = os.path.split(thm["url"])
        #         deps_dir = LEAN4_DEPS_DIR if uses_lean4 else LEAN3_DEPS_DIR
        #         file_path = os.path.join(deps_dir, repo_name, thm["file_path"])

        #     for i, tac in enumerate(thm["traced_tactics"]):
        #         state = format_state(tac["state_before"])
        #         context = Context(
        #             file_path, thm["full_name"], Pos(*thm["start"]), state
        #         )
        #         all_pos_premises = get_all_pos_premises(
        #             tac["annotated_tactic"], self.corpus
        #         )

        #         if self.is_train:
        #             # In training, we ignore tactics that do not have any premises.
        #             for pos_premise in all_pos_premises:
        #                 data.append(
        #                     {
        #                         "url": thm["url"],
        #                         "commit": thm["commit"],
        #                         "file_path": thm["file_path"],
        #                         "full_name": thm["full_name"],
        #                         "start": thm["start"],
        #                         "tactic_idx": i,
        #                         "context": context,
        #                         "pos_premise": pos_premise,
        #                         "all_pos_premises": all_pos_premises,
        #                     }
        #                 )
        #         else:
        #             data.append(
        #                 {
        #                     "url": thm["url"],
        #                     "commit": thm["commit"],
        #                     "file_path": thm["file_path"],
        #                     "full_name": thm["full_name"],
        #                     "start": thm["start"],
        #                     "tactic_idx": i,
        #                     "context": context,
        #                     "all_pos_premises": all_pos_premises,
        #                 }
        #             )

        logger.info(f"Loaded {len(data)} examples.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if not self.is_train:
            return self.data[idx]

        # In-file negatives + random negatives from all accessible premises.
        ex = deepcopy(self.data[idx])
        premises_in_file = []
        premises_outside_file = []


        # TODO: implement negative sampling from the context.
        # for p in self.corpus.get_premises(ex["context"].path):
        #     if p == ex["pos_premise"]:
        #         continue
        #     if p.end < ex["context"].theorem_pos:
        #         if ex["pos_premise"].path == ex["context"].path:
        #             premises_in_file.append(p)
        #         else:
        #             premises_outside_file.append(p)

        # for p in self.corpus.transitive_dep_graph.successors(ex["context"].path):
        #     if p == ex["pos_premise"].path:
        #         premises_in_file += [
        #             _p for _p in self.corpus.get_premises(p) if _p != ex["pos_premise"]
        #         ]
        #     else:
        #         premises_outside_file += self.corpus.get_premises(p)
        # num_in_file_negatives = min(len(premises_in_file), self.num_in_file_negatives)

        premises_in_file = [] # TODO: correctly instantiate!
        premises_outside_file = self.all_premises

        num_in_file_negatives = min(len(premises_in_file), self.num_in_file_negatives)

        ex["neg_premises"] = random.sample(
            premises_in_file, num_in_file_negatives
        ) + random.sample(
            premises_outside_file, self.num_negatives - num_in_file_negatives
        )
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        # Tokenize the context.
        context = [ex["context"] for ex in examples]
        tokenized_context = self.tokenizer(
            [c.serialize_name_type() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask

        # Tokenize the label and premises.
        if self.is_train:
            pos_premise = [ex["pos_premise"] for ex in examples]
            tokenized_pos_premise = self.tokenizer(
                # [p.serialize() for p in pos_premise],
                [p for p in pos_premise],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            batch["pos_premise"] = pos_premise
            batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
            batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            for j in range(batch_size):
                all_pos_premises = examples[j]["all_pos_premises"]
                for kpos in range(batch_size):
                    # pos_premise_k = examples[k]["pos_premise"]
                    label[j, kpos] = 1.0
                
                for kneg in range (batch_size * self.num_negatives):
                    (bix, nix) = divmod(kneg, self.num_negatives) # kneg // self.num_negatives, kneg % self.num_negatives
                    # should use `modrem` ?
                    neg_premise = examples[bix]["neg_premises"][nix]
                    # it might accidentally be included, test for that hypothesis...
                    label[j, kneg] = float(neg_premise in all_pos_premises)
            
            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.tokenizer(
                    # [p.serialize() for p in neg_premise],
                    [p for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    data_path : str
    all_premises: List[str]

    def __init__(
        self,
        data_path: str,
        # corpus_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.corpus = Corpus(corpus_path)
        # self.corpus = Corpus(data_path)
        # metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
        # repo = LeanGitRepo(**metadata["from_repo"])
        # self.uses_lean4 = repo.uses_lean4

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.all_premises = set()
        if stage in (None, "fit"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.jsonl")],
                # self.uses_lean4,
                # self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )
            self.all_premises.update(self.ds_train.all_premises)

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                [os.path.join(self.data_path, "validate.jsonl")],
                # self.uses_lean4,
                # self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )
            self.all_premises.update(self.ds_val.all_premises)

        if stage in (None, "fit", "predict"):
            # TODO: not sure this is right. Actually only take whatever the user asks us to take?
            # Actually, we should probably only take 'test', as `validate` is called per epoch to
            # decide on the best model.
            self.ds_pred = RetrievalDataset(
                [os.path.join(self.data_path, "test.jsonl")],
                # [
                #     os.path.join(self.data_path, f"{split}.jsonl")
                #     for split in ("train", "validate", "test")
                # ],
                # self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )
            self.all_premises.update(self.ds_pred.all_premises)
        self.all_premises = list(self.all_premises)
        print(f"all premises length: {len(self.all_premises)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
