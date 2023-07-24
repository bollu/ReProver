"""Ligihtning module for the premise retriever."""
import os
import math
import torch
import pickle
import numpy as np
from tqdm import tqdm
# from lean_dojo import Pos
# from commonfstar import Pos
from loguru import logger
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Union
from transformers import T5EncoderModel, AutoTokenizer
import pudb
import json
import pprint
import dataclasses
from retrievalfstar.datamodule import RetrievalDataModule
from dataclasses import dataclass, field

pp = pprint.PrettyPrinter(indent=2)


from commonfstar import (
    get_optimizers,
    load_checkpoint,
    zip_strict,
    cpu_checkpointing_enabled,
)


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

torch.set_float32_matmul_precision("medium")


# TODO: where is `self.trainer` even from?
class PremiseRetriever(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float,
        warmup_steps: int,
        num_retrieved: int,
        max_seq_len: int,
        d_embed: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_retrieved = num_retrieved
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.linear_context = torch.nn.Linear(self.encoder.config.d_model, d_embed)
        self.linear_premise = torch.nn.Linear(self.encoder.config.d_model, d_embed)
        self.embeddings_staled = True

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "PremiseRetriever":
        return load_checkpoint(cls, ckpt_path, device, freeze)

    @property
    def embedding_size(self) -> int:
        """Return the size of the feature vector produced by ``encoder``."""
        return self.encoder.config.hidden_size

    def _encode(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Encode a premise or a context into a feature vector."""
        if cpu_checkpointing_enabled(self):
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.encoder, input_ids, attention_mask, use_reentrant=False
            )[0]
        else:
            hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        # Masked average.
        lens = attention_mask.sum(dim=1)
        features = (hidden_states * attention_mask.unsqueeze(2)).sum(
            dim=1
        ) / lens.unsqueeze(1)

        # Normalize the feature vector to have unit norm.
        # TODO: consider adding linear layer here?
        return F.normalize(features, dim=1)


    def forward(
        self,
        context_ids: torch.LongTensor,
        context_mask: torch.LongTensor,
        pos_premise_ids: torch.LongTensor,
        pos_premise_mask: torch.LongTensor,
        neg_premises_ids: torch.LongTensor,
        neg_premises_mask: torch.LongTensor,
        label: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute the contrastive loss for premise retrieval.
        The DataLoder performs tokenization, and the model runs
        the encoder decoder model onto the tokenized stream.
        """
        # Encode the query and positive/negative documents.
        context_emb =   self.linear_context(self._encode(context_ids, context_mask))
        pos_premise_emb = self.linear_premise(self._encode(pos_premise_ids, pos_premise_mask))
        neg_premise_embs = [
            self.linear_premise(self._encode(ids, mask))
            for ids, mask in zip_strict(neg_premises_ids, neg_premises_mask)
        ]
        all_premise_embs = torch.cat([pos_premise_emb, *neg_premise_embs], dim=0)

        # Cosine similarities for unit-norm vectors are just inner products.
        similarity = torch.mm(context_emb, all_premise_embs.t())
        assert -1 <= similarity.min() <= similarity.max() <= 1
        loss = F.mse_loss(similarity, label)
        return loss

    ############
    # Training #
    ############

    def on_fit_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
            logger.info(f"Logging to {self.trainer.log_dir}")

        # print(self.trainer.datamodule)
        # pudb.set_trace()
        # self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        loss = self(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )
        self.log(
            "loss_train", loss, on_epoch=True, sync_dist=True, batch_size=len(batch)
        )
        return loss

    def on_train_batch_end(self, outputs, batch, _) -> None:
        """Mark the embeddings as staled after a training batch."""
        self.embeddings_staled = True

    def configure_optimizers(self) -> Dict[str, Any]:
        return get_optimizers(
            self.parameters(), self.trainer, self.lr, self.warmup_steps
        )

    ##############
    # Validation #
    ##############

    @torch.no_grad()
    def reindex_corpus(self, batch_size: int) -> None:
        """Re-index the retrieval corpus using the up-to-date encoder."""
        if not self.embeddings_staled:
            return
        all_premise_names = self.trainer.datamodule.corpus.premise_names
        logger.info("Re-indexing the retrieval corpus")
        logger.info(f"trainer: {pp.pformat(self.trainer)}")
        self.corpus_embeddings = torch.zeros(
            len(all_premise_names),
            self.embedding_size,
            dtype=self.encoder.dtype,
            device=self.device,
        )

        print(f"all premises length: '{len(all_premise_names)}'")
        for i in tqdm(range(0, len(all_premise_names), batch_size)):
            batch_premises = [self.trainer.datamodule.corpus.get_premise_embed_str_for_name(name) for name in all_premise_names[i : i + batch_size] ]
            tokenized_premises = self.tokenizer(
                [p for p in batch_premises], # TODO: make this `Premise` object
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.corpus_embeddings[i : i + batch_size] = self._encode(
                tokenized_premises.input_ids, tokenized_premises.attention_mask
            )

        self.embeddings_staled = False

    def on_validation_start(self) -> None:
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)

    def get_nearest_premises(self,
        datamodule : RetrievalDataModule,
        premise_embeddings: torch.FloatTensor,
        batch_context_names: List[str],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[float]]]:
        """Perform a batch of nearest neighbour search. and returns the names of the closest premises"""
        similarities = self.linear_context(batch_context_emb) @ self.linear_premise(premise_embeddings).t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context_names]
        scores = [[] for _ in batch_context_names]

        for j, (ctx_name, idxs) in enumerate(zip(batch_context_names, idxs_batch)):
            for i in idxs:
                premise_name = datamodule.corpus.all_premise_names[i]
                # if p in accessible_premises:
                if datamodule.corpus.occurs_before_loc_by_name(premise_name, ctx_name):
                    results[j].append(premise_name)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
        return results, scores


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Retrieve premises and calculate metrics such as Recall@K and MRR."""
        # Retrieval.
        # Who builds `batch`? There should be a `collate` that gets called
        # by someone to build `batch`
        context_emb = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        # TODO: this depends on `corpus` because it filters to only find those
        # premises which are currently reachable from the current corpus.
        # It feels perverse to include this in the code.
        # Feels like it artificially boosts the evaluation of the model (?)

        # retrieved_premises, _ = self.corpus.get_nearest_premises(
        #     self.corpus_embeddings,
        #     batch["context"],
        #     context_emb,
        #     self.num_retrieved,
        # )

        retrieved_premises, _ = self.get_nearest_premises(
            self.trainer.datamodule,
            self.corpus_embeddings,
            batch["context_name"],
            context_emb,
            self.num_retrieved
        )

        # Evaluation & logging.
        recall = [[] for _ in range(self.num_retrieved)]
        MRR = []
        num_with_premises = 0
        tb = self.logger.experiment
        # pp.pprint(f"batch: {batch.keys()}")
        for i, (all_pos_premise_names, premises) in enumerate(
            zip_strict(batch["pos_premise_name"], retrieved_premises)
        ):
            # Only log the first example in the batch.
            if i == 0:
                # msg_gt = "\n\n".join([p.serialize() for p in all_pos_premise_names])
                msg_gt = "\n\n".join([p for p in all_pos_premise_names])
                # msg_retrieved = "\n\n".join(
                #     [f"{j}. {p.serialize()}" for j, p in enumerate(premises)]
                # )
                msg_retrieved = "\n\n".join(
                    [f"{j}. {p}" for j, p in enumerate(premises)]
                )

                TP = len(set(premises).intersection(all_pos_premise_names))
                if len(all_pos_premise_names) == 0:
                    r = math.nan
                else:
                    r = float(TP) / len(all_pos_premise_names)
                msg = f"Recall@{self.num_retrieved}: {r}\n\nGround truth:\n\n```\n{msg_gt}\n```\n\nRetrieved:\n\n```\n{msg_retrieved}\n```"
                tb.add_text(f"premises_val", msg, self.global_step)

            all_pos_premise_names = set(all_pos_premise_names)
            if len(all_pos_premise_names) == 0:
                continue
            else:
                num_with_premises += 1
            first_match_found = False

            for j in range(self.num_retrieved):
                TP = len(all_pos_premise_names.intersection(premises[: (j + 1)]))
                recall[j].append(float(TP) / len(all_pos_premise_names))
                if premises[j] in all_pos_premise_names and not first_match_found:
                    MRR.append(1.0 / (j + 1))
                    first_match_found = True
            if not first_match_found:
                MRR.append(0.0)

        recall = [100 * np.mean(_) for _ in recall]

        for j in range(self.num_retrieved):
            self.log(
                f"Recall@{j+1}_val",
                recall[j],
                on_epoch=True,
                sync_dist=True,
                batch_size=num_with_premises,
            )

        self.log(
            "MRR",
            np.mean(MRR),
            on_epoch=True,
            sync_dist=True,
            batch_size=num_with_premises,
        )

    ##############
    # Testing    #
    ##############

    def on_test_start(self) -> None:
        # self.corpus = self.trainer.datamodule.corpus
        self.corpus_embeddings = None
        self.embeddings_staled = True
        logger.info(f"Evaluating on {self.trainer.datamodule.data_path}")
        self.reindex_corpus(self.trainer.datamodule.eval_batch_size)
        self.stats = dict()

    def test_step(self, batch: Dict[str, Any], batch_idx : int, dataloader_idx : int):
        if dataloader_idx not in self.stats:
            self.stats[dataloader_idx] = TestStatisticsCollator()
        collator = self.stats[dataloader_idx]
        # recall that everything is batched
        context_emb_batched = self._encode(batch["context_ids"], batch["context_mask"])
        assert not self.embeddings_staled
        # logger.info(f"batch.keys(): '{batch.keys()}', corpus embeddings: '{self.corpus_embeddings.shape}', num_retrieved: '{self.num_retrieved}'")
    
        assert self.corpus_embeddings is not None
        # ALL_POS_PREMISES: BATCHSIZE x <ragged>
        all_pos_premises_batched = batch["all_pos_premises"]
        # RETRIEVED_PREMISES: BATCHSIZE x NUM_RETRIEVED
        retrieved_premises_batched, scores_batched = self.get_nearest_premises(
            self.trainer.datamodule,
            self.corpus_embeddings,
            batch["context"],
            context_emb_batched,
            self.num_retrieved,
        )

        for bix in range(len(all_pos_premises_batched)):
            all_pos_premises = all_pos_premises_batched[bix]
            all_pos_premises_set = set(all_pos_premises)
            retrieved_premises = retrieved_premises_batched[bix]

            TP1 = retrieved_premises[0] in all_pos_premises
            R1 = float(TP1) / len(all_pos_premises) * 100.0
            collator.R1s.append(R1)
            TP10 = len(all_pos_premises_set.intersection(retrieved_premises[:10]))
            R10 = float(TP10) / len(all_pos_premises) * 100.0
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

            collator.test_step_outputs.append({
                "context": dataclasses.asdict(batch["context"][bix]),
                "all_pos_premises": all_pos_premises,
                "retrieved_premises": retrieved_premises,
                "scores": scores_batched[bix],
                "R1": R1,
                "R10": R10,
                "RR": RR,
                "AP": AP,
                "NDCG": NDCG,
                "K_at_full_recall": K_at_full_recall,
                "K_percent_at_full_recall": K_percent_at_full_recall,
            })

        # pp.pprint(f"batch keys: '{batch.keys()}'")

        # for (
        #     # url,
        #     # commit,
        #     # file_path,
        #     # full_name,
        #     # start,
        #     # tactic_idx,
        #     ctx,
        #     pos_premises,
        #     premises,
        #     s,
        # ) in zip_strict(
        #     # batch["url"],
        #     # batch["commit"],
        #     # batch["file_path"],
        #     # batch["full_name"],
        #     # batch["start"],
        #     # batch["tactic_idx"],
        #     batch["context"],
        #     batch["all_pos_premises"],
        #     retrieved_premises,
        #     scores,
        # ):
        #     self.test_step_outputs.append(
        #         {
        #             # "url": url,
        #             # "commit": commit,
        #             # "file_path": file_path,
        #             # "full_name": full_name,
        #             # "start": start,
        #             # "tactic_idx": tactic_idx,
        #             "context": ctx,
        #             "all_pos_premises": pos_premises,
        #             "retrieved_premises": premises,
        #             "scores": s,
        #         }
        #     )

    def on_test_epoch_end(self) -> None:
        for ix in self.stats.keys():
            collator = self.stats[ix]
            R1 = np.mean(collator.R1s)
            R10 = np.mean(collator.R10s)
            MRR = np.mean(collator.RRs)
            MAP = np.mean(collator.APs)
            NDCG = np.mean(collator.NDCGs)
            K_at_full_recall = np.nanmean(collator.Ks_at_full_recall)
            K_percent_at_full_recall = np.nanmean(collator.Ks_percent_at_full_recall)
            num_no_full_recall = np.count_nonzero(np.isnan(collator.Ks_at_full_recall))
            percent_no_full_recall = num_no_full_recall / len(collator.Ks_at_full_recall)
            name = ["train", "val", "test"][ix] # TODO: pick this up from the data loader.
            logger.info(f"name={name} R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}, MAP = {MAP}, NDCG = {NDCG}")
            logger.info(f"  K for full recall = {K_at_full_recall}, %K for full recall = {K_percent_at_full_recall}, #no full recall = {num_no_full_recall}, %no full recall: {percent_no_full_recall}")

            if self.trainer.log_dir is not None:
                path = os.path.join(self.trainer.log_dir, f"test_output_{name}.json")
                with open(path, "w") as of:
                    json.dump({
                        "predict_steps": collator.test_step_outputs,
                        "R1": R1,
                        "R10": R10,
                        "MRR": MRR,
                        "MAP": MAP,
                        "NDCG": NDCG,
                        "K_at_full_recall": K_at_full_recall,
                        "K_percent_at_full_recall": K_percent_at_full_recall,
                        "num_no_full_recall": num_no_full_recall,
                        "percent_no_full_recall": percent_no_full_recall,
                    }, of, indent=2)
                logger.info(f"Retrieval predictions saved to {path}")

