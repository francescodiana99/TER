import typer
from pathlib import Path
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
from thinc.api import Config
import wandb
import spacy


def main(default_config: Path, output_path: Path):
    loaded_local_config = util.load_config(default_config)
    with wandb.init() as run:
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        spacy.require_gpu(0)
        nlp = init_nlp(merged_config)
        #overrides = {"paths.train": "./cv/train_0.spacy", "paths.dev": "./cv/eval_0.spacy"}
        train(nlp, output_path, use_gpu=True)


if __name__ == "__main__":
    typer.run(main)
