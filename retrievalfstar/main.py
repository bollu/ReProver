"""Script for training the premise retriever.
"""
import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI
import pretty_errors

from retrievalfstar.model import PremiseRetriever
from retrievalfstar.datamodule import RetrievalDataModule


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    pretty_errors.configure()
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    # import pudb; pudb.set_trace()
    main()
