#!/usr/bin/env python3
# sine selection algorithm implemented from [1]
# [1] Sine Qua Non for Large Theory Reasoning by Krytof et. al
import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI
import pretty_errors
from retrievalfstar.datamodule import Corpus, RetrievalDataModule)


class SineRetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        corpus : Corpus,
        num_negatives: int,
        num_in_file_negatives: int,
        max_seq_len: int,
        tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
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
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        out_data = []
        logger.info(f"Loading data from '{data_path}'")
        in_data = RetrievalDataset.load_json_or_jsonl(data_path)
        for (i, datum) in tqdm(enumerate(in_data)):
            if i == 0:
                print(f"loading datum with keys '{datum.keys()}'")
            print(f"loading '{i}'th data from '{data_path}'. keys: '{datum.keys()}'")
            if not self.corpus.has_definition_for_name(datum["name"]):
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
                    [p for p in datum["premises"] if p not in self.corpus.get_ctx_embed_str_for_name(datum["name"])]
            if not all_pos_premise_names:
                continue # skip this def cause it has zero premises
            know_all_premise_defs = all([self.corpus.has_definition_for_name(p) for p in all_pos_premise_names])
            if not know_all_premise_defs:
                continue # skip this def cause its premise does not occur
            for pos_premise_name in all_pos_premise_names:
                # check that the premise occurs earlier than the context.
                if not self.corpus.occurs_before_loc_by_name(datum["name"], pos_premise_name):
                    logger.error(f"premise '{pos_premise_name}' in file '{self.corpus.get_loc_for_name(pos_premise_name)}'")
                    logger.error(f"^^ does not occur before context '{datum['name']}' in file '{self.corpus.get_loc_for_name(datum['name'])}'")
                assert self.corpus.occurs_before_loc_by_name(datum["name"], pos_premise_name)
                out_data.append({
                    "context_name": datum["name"],
                    "pos_premise_name": pos_premise_name,
                    "all_pos_premise_names": all_pos_premise_names,
                })

        logger.info(f"Loaded '{len(out_data)}' examples.")
        return out_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def collate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """collate and tokenize data returned by self.__getitem__ to build a batch"""
        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]
        return batch


class SineRetrievalDataModule(pl.LightningDataModule):
    data_path : str
    corpus_path : str
    import_graph_path : str
    corpus : Corpus
    
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        import_graph_path : str,
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
        self.corpus_path = corpus_path
        self.import_graph_path = import_graph_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.corpus = Corpus(corpus_path=corpus_path, import_graph_path=import_graph_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "test"):
            self.ds_train = RetrievalDataset(
                [os.path.join(self.data_path, "train.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=True,
            )

        if stage in (None, "fit", "validate", "test"):
            self.ds_val = RetrievalDataset(
                [os.path.join(self.data_path, "validate.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

        if stage in (None, "fit", "test"):
            # TODO: not sure this is right. Actually only take whatever the user asks us to take?
            # Actually, we should probably only take 'test', as `validate` is called per epoch to
            # decide on the best model.
            self.ds_test = RetrievalDataset(
                [os.path.join(self.data_path, "test.json")],
                # [
                #     os.path.join(self.data_path, f"{split}.jsonl")
                #     for split in ("train", "validate", "test")
                # ],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
            )

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

    def test_dataloader(self) -> List[DataLoader]:
        out_dses = []
        for ds in [self.ds_train, self.ds_val, self.ds_test]:
            out_dses.append(DataLoader(
                ds,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=ds.collate,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                ))
        return out_dses



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
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

    @classmethod
    def load(cls, ckpt_path: str, device, freeze: bool) -> "PremiseRetriever":
        pass

    def training_step(self, batch: Dict[str, Any], _) -> torch.Tensor:
        return None # model that needs no training

    ##############
    # Validation #
    ##############
    def get_nearest_premises(self,
        datamodule : RetrievalDataModule,
        batch_context_names: List[str],
        k: int,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Perform a batch of nearest neighbour search. and returns the names of the closest premises"""
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context_names]
        scores = [[] for _ in batch_context_names]

        for j, (ctx_name, idxs) in enumerate(zip(batch_context_names, idxs_batch)):
            for i in idxs:
                premise_name = datamodule.corpus.all_premise_names[i]
                # if p in accessible_premises:
                if datamodule.corpus.occurs_before_loc_by_name(ctx_name, premise_name):
                    results[j].append(premise_name)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break

        return results, scores


    ##############
    # Testing    #
    ##############

    def on_test_start(self) -> None:
        self.embeddings_staled = True

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


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    pretty_errors.configure()
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(SinePremiseRetriever, SineRetrievalDataModule)
    logger.info("Configuration: \n", cli.config)

if __name__ == "__main__":
    pass
