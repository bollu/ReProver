import os
import re
import sys
import json
import random
import torch
import tempfile
import networkx as nx
from loguru import logger
# from lean_dojo import Pos
import pytorch_lightning as pl
from dataclasses import dataclass, field
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from transformers import get_cosine_schedule_with_warmup
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from typing import Optional, List, Dict, Any, Tuple, Generator
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


@dataclass(eq=True, unsafe_hash=True)
class Pos:
    """Position in source files.

    We use 1-index to keep it consistent with code editors such as Visual Studio Code.
    """

    line_nb: int
    """Line number
    """

    column_nb: int
    """Column number
    """

    @classmethod
    def from_str(cls, s: str) -> "Pos":
        """Construct a :class:`Pos` object from its string representation, e.g., :code:`"(323, 1109)"`."""
        assert s.startswith("(") and s.endswith(
            ")"
        ), f"Invalid string representation of a position: {s}"
        line, column = s[1:-1].split(",")
        line_nb = int(line)
        column_nb = int(column)
        return cls(line_nb, column_nb)

    def __iter__(self) -> Generator[int, None, None]:
        yield self.line_nb
        yield self.column_nb

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __lt__(self, other):
        return self.line_nb < other.line_nb or (
            self.line_nb == other.line_nb and self.column_nb < other.column_nb
        )

    def __le__(self, other):
        return self < other or self == other
Example = Dict[str, Any]
Batch = Dict[str, Any]

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"

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

def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")

@dataclass(unsafe_hash=True)
class Context:
    """Contexts are "queries" in our retrieval setup."""

    name: str
    type_ : str
    definition : str

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.type_, str)
        assert isinstance(self.definition, str)
        # assert isinstance(self.theorem_pos, Pos)

    def serialize(self) -> str:
        """Serialize the context into a string for Transformers."""
        return "[n[" + self.name + "] t[" + self.type_ + "] d[" + self.definition + "]]" 


@dataclass(unsafe_hash=True)
class Premise:
    """
    Premises are "documents" in our retrieval setup.
    Created by `File`.
    """

    full_name: str
    """Fully qualified name.
    """

    # start: Pos = field(repr=False)
    """Start position of the premise's definition in the ``*.lean`` file.
    """

    # end: Pos = field(repr=False, compare=False)
    """End position of the premise's definition in the ``*.lean`` file.
    """

    definition: str = field
    # code: str = field(compare=False)
    """Raw, human-written code for defining the premise.
    """

    # type_ : str
    """
    type signature of the premise
    """

    # premises_for_proof: List[str]
    """
    other premises used in the proof of this premise.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.path, str)
        assert isinstance(self.full_name, str)
        assert (
            isinstance(self.start, Pos)
            and isinstance(self.end, Pos)
            and self.start <= self.end
        )
        assert isinstance(self.definition, str) and self.code != ""

    def serialize(self) -> str:
        """Serialize the premise into a string for Transformers."""
        annot_full_name = f"{MARK_START_SYMBOL}{self.full_name}{MARK_END_SYMBOL}"
        code = self.code.replace(f"_root_.{self.full_name}", annot_full_name)
        fields = self.full_name.split(".")

        for i in range(len(fields)):
            prefix = ".".join(fields[i:])
            new_code = re.sub(f"(?<=\s)«?{prefix}»?", annot_full_name, code)
            if new_code != code:
                code = new_code
                break

        return code


class PremiseSet:
    """A set of premises indexed by their paths and full names."""

    # path2premises: Dict[str, Dict[str, Premise]]
    premises: Dict[str, Premise]
    
    def __init__(self) -> None:
        self.premises = {}

    def __iter__(self) -> Generator[Premise, None, None]:
        for p in self.premises.values():
            yield p

    def add(self, p: Premise) -> None:
        self.premises[p.full_name] = p

    def update(self, premises: List[Premise]) -> None:
        for p in premises:
            self.add(p)

    def __contains__(self, p: Premise) -> bool:
        return (
           p.full_name in self.path2premises[p.path]
        )

    def __len__(self) -> int:
        return len(self.premises)


@dataclass(frozen=True)
class File:
    """A file defines 0 or multiple premises."""

    path: str
    """Path of the ``*.lean`` file.
    """

    premises: List[Premise]
    """A list of premises defined in this file.
    """

    @classmethod
    def from_data(cls, file_data: Dict[str, Any]) -> "File":
        """Construct a :class:`File` object from ``file_data``."""
        # `file_data` is the `jsonl`
        # TODO: read this, who even constructs a File?
        # Answer: `Corpus` creates a `File`.
        path = file_data["path"]
        premises = []
        for p in file_data["premises"]:
            if "user__.n" in p["full_name"] or p["code"] == "":
                # Ignore ill-formed premises (often due to errors in ASTs).
                continue
            full_name = p["full_name"]
            if full_name.startswith("[") and full_name.endswith("]"):
                # Ignore mutual definitions.
                continue
            premises.append(
                Premise(
                    path, p["full_name"], Pos(*p["start"]), Pos(*p["end"]), p["code"]
                )
            )
        return cls(path, premises)

    @property
    def is_empty(self) -> bool:
        """Check whether the file contains no premise."""
        return self.premises == []


class Corpus:
    """Our retrieval corpus is a DAG of files. Each file consists of
    premises (theorems, definitoins, etc.) that can be retrieved.
    """

    # transitive_dep_graph: nx.DiGraph
    """Transitive closure of the dependency graph among files.
    There is an edge from file X to Y iff X import Y (directly or indirectly).
    """

    all_premises: List[Premise]
    """All premises in the entire corpus.
    """

    def __init__(self, jsonl_path: str) -> None:
        # """Construct a :class:`Corpus` object from a ``corpus.jsonl`` data file."""
        """Construct a :class:`Corpus` object from a ``data.jsonl`` data file."""
        dep_graph = nx.DiGraph()
        self.all_premises = []

        logger.info(f"Building the corpus from {jsonl_path}")

        # vvv below code is necessary where there is a corpus.
        # we have but a single file.
        # for line in open(jsonl_path):
            # file_data = json.loads(line)
            # file_name = file_data["file_name"]
            # assert not dep_graph.has_node(path)
            # file = File.from_data(file_data)

            # dep_graph.add_node(path, file=file)
            # self.all_premises.extend(file.premises)

            # for p in file_data["imports"]:
                # assert dep_graph.has_node(p)
                # dep_graph.add_edge(path, p)

        # assert nx.is_directed_acyclic_graph(dep_graph)
        # self.transitive_dep_graph = nx.transitive_closure_dag(dep_graph)

        self.imported_premises_cache = {}
        self.fill_cache()

    def _get_file(self, path: str) -> File:
        return self.transitive_dep_graph.nodes[path]["file"]

    def __len__(self) -> int:
        return len(self.all_premises)

    def __contains__(self, path: str) -> bool:
        return path in self.transitive_dep_graph

    def __getitem__(self, idx: int) -> Premise:
        return self.all_premises[idx]

    @property
    def files(self) -> List[File]:
        return [self._get_file(p) for p in self.transitive_dep_graph.nodes]

    @property
    def num_files(self) -> int:
        return len(self.files)

    def get_dependencies(self, path: str) -> List[str]:
        """Return a list of (direct and indirect) dependencies of the file ``path``."""
        return list(self.transitive_dep_graph.successors(path))

    def get_premises(self, path: str) -> List[Premise]:
        """Return a list of premises defined in the file ``path``."""
        return self._get_file(path).premises

    def num_premises(self, path: str) -> int:
        """Return the number of premises defined in the file ``path``."""
        return len(self.get_premises(path))

    def locate_premise(self, path: str, pos: Pos) -> Optional[Premise]:
        """Return a premise at position ``pos`` in file ``path``.

        Return None if no such premise can be found.
        """
        for p in self.get_premises(path):
            assert p.path == path
            if p.start <= pos <= p.end:
                return p
        return None

    def fill_cache(self) -> None:
        for path in self.transitive_dep_graph.nodes:
            self._get_imported_premises(path)

    def _get_imported_premises(self, path: str) -> List[Premise]:
        """Return a list of premises imported in file ``path``. The result is cached."""
        premises = self.imported_premises_cache.get(path, None)
        if premises is not None:
            return premises

        premises = []
        for p in self.transitive_dep_graph.successors(path):
            premises.extend(self._get_file(p).premises)
        self.imported_premises_cache[path] = premises
        return premises

    def get_accessible_premises(self, path: str, pos: Pos) -> PremiseSet:
        """Return the set of premises accessible at position ``pos`` in file ``path``,
        i.e., all premises defined in the (transitively) imported files or earlier in the same file.
        """
        premises = PremiseSet()
        for p in self.get_premises(path):
            if p.end <= pos:
                premises.add(p)
        premises.update(self._get_imported_premises(path))
        return premises

    def get_accessible_premise_indexes(self, path: str, pos: Pos) -> List[int]:
        return [
            i
            for i, p in enumerate(self.all_premises)
            if (p.path == path and p.end <= pos)
            or self.transitive_dep_graph.has_edge(path, p.path)
        ]

    def get_nearest_premises(
        self,
        premise_embeddings: torch.FloatTensor,
        batch_context: List[Context],
        batch_context_emb: torch.Tensor,
        k: int,
    ) -> Tuple[List[List[Premise]], List[List[float]]]:
        """Perform a batch of nearest neighbour search."""
        similarities = batch_context_emb @ premise_embeddings.t()
        idxs_batch = similarities.argsort(dim=1, descending=True).tolist()
        results = [[] for _ in batch_context]
        scores = [[] for _ in batch_context]

        for j, (ctx, idxs) in enumerate(zip(batch_context, idxs_batch)):
            accessible_premises = self.get_accessible_premises(
                ctx.path, ctx.theorem_pos
            )
            for i in idxs:
                p = self.all_premises[i]
                if p in accessible_premises:
                    results[j].append(p)
                    scores[j].append(similarities[j, i].item())
                    if len(results[j]) >= k:
                        break
            else:
                raise ValueError

        return results, scores


@dataclass(frozen=True)
class IndexedCorpus:
    """A corpus with premise embeddings."""

    corpus: Corpus
    embeddings: torch.FloatTensor

    def __post_init__(self):
        assert self.embeddings.device == torch.device("cpu")
        assert len(self.embeddings) == len(self.corpus)


def get_all_pos_premises(annot_tac, corpus: Corpus) -> List[Premise]:
    """Return a list of all premises that are used in the tactic ``annot_tac``."""
    _, provenances = annot_tac
    all_pos_premises = set()

    for prov in provenances:
        def_path = prov["def_path"]
        p = corpus.locate_premise(def_path, Pos(*prov["def_pos"]))
        if p is not None:
            all_pos_premises.add(p)

    return list(all_pos_premises)


_SPACES_REGEX = re.compile(r"\s+", re.DOTALL)


def normalize_spaces(s: str) -> str:
    """Repalce any consecutive block of whitespace characters in ``s`` with a single whitespace."""
    return _SPACES_REGEX.sub(" ", s).strip()


def format_tactic(annot_tac: str, provenances, normalize: bool) -> str:
    """Use full names for the all <a>...</a>."""
    if normalize:
        annot_tac = normalize_spaces(annot_tac)
    if len(provenances) == 0:
        return annot_tac

    tac = ""
    marks = list(re.finditer(r"<a>(?P<ident>.+?)</a>", annot_tac))

    for i, (m, prov) in enumerate(zip_strict(marks, provenances)):
        last_end = marks[i - 1].end() if i > 0 else 0
        tac += annot_tac[last_end : m.start()] + "<a>" + prov["full_name"] + "</a>"

    tac += annot_tac[marks[-1].end() :]
    return tac


def format_state(s: str) -> str:
    m = re.match(r"\d+ goals", s)
    if m is not None:
        return s[m.end() :].strip()
    else:
        return s


def format_augmented_state(
    s: str, premises: List[Premise], max_len: int, p_drop: float
) -> str:
    """Format a state with retrieved premises and drop some of them with probability ``p_drop``."""
    s = format_state(s)

    aug_s = ""
    length = 0
    max_premises_len = max_len - len(bytes(s.encode("utf-8")))

    for p in premises:
        if random.random() < p_drop:
            continue
        p_str = f"{p.serialize()}\n\n"
        l = len(bytes(p_str.encode("utf-8")))
        if length + l > max_premises_len:
            continue
        length += l
        aug_s = p_str + aug_s

    aug_s += s
    return aug_s


def get_optimizers(
    parameters, trainer: pl.Trainer, lr: float, warmup_steps: int
) -> Dict[str, Any]:
    """Return an AdamW optimizer with cosine warmup learning rate schedule."""
    strategy = trainer.strategy

    if isinstance(strategy, DeepSpeedStrategy):
        if "offload_optimizer" in strategy.config["zero_optimization"]:
            logger.info("Optimizing with DeepSpeedCPUAdam")
            optimizer = DeepSpeedCPUAdam(parameters, lr=lr, adamw_mode=True)
        else:
            logger.info("Optimizing with FusedAdam")
            optimizer = FusedAdam(parameters, lr=lr, adam_w_mode=True)
    else:
        logger.info("Optimizing with AdamW")
        optimizer = torch.optim.AdamW(parameters, lr=lr)

    if trainer.max_steps != -1:
        max_steps = trainer.max_steps
    else:
        assert trainer.max_epochs is not None
        max_steps = (
            trainer.max_epochs
            * len(trainer.datamodule.train_dataloader())
            // trainer.accumulate_grad_batches
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _is_deepspeed_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"Checkpoint {path} does not exist.")
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "zero_to_fp32.py"))


def load_checkpoint(model_cls, ckpt_path: str, device, freeze: bool):
    """Handle DeepSpeed checkpoints in model loading."""
    if not _is_deepspeed_checkpoint(ckpt_path):
        model = model_cls.load_from_checkpoint(ckpt_path, strict=False).to(device)
    else:
        with tempfile.TemporaryDirectory() as dirname:
            path = os.path.join(dirname, "lightning.cpkt")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, path)
            model = model_cls.load_from_checkpoint(path, strict=False)
            model = model.to(device)
    if freeze:
        model.freeze()
    return model


def zip_strict(*args):
    assert len(args) > 1 and all(len(args[0]) == len(a) for a in args[1:])
    return zip(*args)


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


def cpu_checkpointing_enabled(pl_module) -> bool:
    try:
        trainer = pl_module.trainer
        return (
            trainer.strategy is not None
            and isinstance(trainer.strategy, DeepSpeedStrategy)
            and trainer.strategy.config["activation_checkpointing"]["cpu_checkpointing"]
        )
    except RuntimeError:
        return False
