"""Script for evaluating the premise retriever."""
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from loguru import logger


def _eval(data, preds_map) -> Tuple[float, float, float]:
    R1 = []
    R10 = []
    MRR = []
    MAP = []
    NDCG = []

    # Ah, interesting, they evaluate on a _different_ thing
    # they evaluate for each piece in the data
    for line in tqdm(data): # data is a jsonl file.
        tnhm = json.loads(line)
        # print(f"thm keys: {thm.keys()}")
        # for 
        # for i, _ in enumerate(thm["traced_tactics"]):
        #     pred = preds_map[
        #         (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
        #     ]
        #     all_pos_premises = set(pred["all_pos_premises"])
        #     if len(all_pos_premises) == 0:
        #         continue

        #     retrieved_premises = pred["retrieved_premises"]
        #     TP1 = retrieved_premises[0] in all_pos_premises
        #     R1.append(float(TP1) / len(all_pos_premises))
        #     TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
        #     R10.append(float(TP10) / len(all_pos_premises))

        #     for j, p in enumerate(retrieved_premises):
        #         if p in all_pos_premises:
        #             MRR.append(1.0 / (j + 1))
        #             break
        #     else:
        #         MRR.append(0.0)
            
        #     AP = 0
        #     DCG = 0
        #     for j, p in enumerate(retrieved_premises):
        #         if p in all_pos_premises:
        #             AP += 1.0 / (j + 1)
        #             DCG += 1.0 / (np.log2(j + 1) if j > 0 else 1)

        #     IDCG = 0
        #     for j in range(len(all_pos_premises)):
        #         IDCG += 1.0 / (np.log2(j + 1) if j > 0 else 1)
        #     AP /= len(all_pos_premises)
        #     MAP.append(AP)
        #     NDCG.append(DCG / IDCG)


    R1 = 100 * np.mean(R1)
    R10 = 100 * np.mean(R10)
    MRR = np.mean(MRR)
    MAP = np.mean(MAP)
    NDCG = np.mean(NDCG)
    return R1, R10, MRR, MAP, NDCG


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the premise retriever."
    )
    parser.add_argument(
        "--preds-file",
        type=str,
        required=True,
        help="Path to the retriever's predictions file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the directory containing the train/val/test splits.",
    )
    args = parser.parse_args()
    logger.info(args)

    logger.info(f"Loading predictions from {args.preds_file}")
    preds = pickle.load(open(args.preds_file, "rb"))
    # dict_keys(['context', 'all_pos_premises', 'retrieved_premises', 'scores'])
    ps = set()
    # for p in preds:
    #     print(p["context"].name)
    #     ps.add(p["context"].name)
    # print(f"all predictions: {ps}")

    preds_map = {
        p["context"].name : p
        for p in preds
    }
    print(f"#preds in preds_map : {len(preds_map)} | #preds: {len(preds)}")
    # TODO: figure out what the hell is going on here.
    # assert len(preds) == len(preds_map), "Duplicate predictions found!"

    for split in ("train", "validate", "test"):
        data_path = os.path.join(args.data_path, f"{split}.jsonl")
        data = open(data_path).readlines()
        logger.info(f"Evaluating on {data_path}")
        R1, R10, MRR, MAP, NDCG = _eval(data, preds_map)
        logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}, MAP = {MAP}, NDCG = {NDCG}")


if __name__ == "__main__":
    main()
