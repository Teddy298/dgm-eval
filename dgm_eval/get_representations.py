import os
import pathlib
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
import torch
import time
import pathlib

from .__main__ import (
    get_device_and_num_workers,
    get_inception_scores,
    get_dataloader_from_path,
    compute_representations,
)
from .dataloaders import get_dataloader
from .heatmaps import visualize_heatmaps
from .helpers import get_last_directory
from .metrics import *
from .models import MODELS, InceptionEncoder, load_encoder
from .representations import get_representations, load_reps_from_path, save_outputs
from . import __main__
from pathlib import Path

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--model",
    type=str,
    default="dinov2",
    choices=MODELS.keys(),
    help="Model to use for generating feature representations.",
)

parser.add_argument(
    "--folder_path",
    type=str,
    default=None,
    help="Path to the main folder containing sets of folders.",
)

parser.add_argument(
    "-bs", "--batch_size", type=int, default=50, help="Batch size to use"
)

parser.add_argument(
    "--num-workers",
    type=int,
    help="Number of processes to use for data loading. "
    "Defaults to `min(8, num_cpus)`",
)

parser.add_argument(
    "--device", type=str, default=None, help="Device to use. Like cuda, cuda:0 or cpu"
)

parser.add_argument(
    "--arch",
    type=str,
    default=None,
    help="Model architecture. If none specified use default specified in model class",
)


parser.set_defaults(load=True)

parser.add_argument("--seed", type=int, default=13579, help="Random seed")


def get_relevant_paths(main_folder_path, key_folder="interpolation"):
    return [x[0] for x in os.walk(main_folder_path) if key_folder in x[0]]


def main():
    output_dir = None
    args = parser.parse_args()

    main_folder_path = Path(args.folder_path)

    relevant_paths = get_relevant_paths(main_folder_path)


    print(relevant_paths)
    return

    new_folder = f"{main_folder_path}_reps"


    # wjezdza sciezka do glownego folderu
    # tam mam te wszystkie GANY
    # wchodzac do GANA spodziewam sie 0, 1, batches i interpolations
    # wylistowac foldery interpolations
    # idac po kazdym folderze przerobic representacje
    # i zapisac do odpowiednika tylko ze, datasets/<model>/<GAN>/<interpolations><X%train>
    # na poczatku skrypt do listowania tych folderow ktore chce
    parent_folder = main_folder_path.parent.absolute()



    os.makedirs("path/to/directory", exist_ok=True)

    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    IS_scores = None
    if "is" in args.metrics and args.model == "inception":
        # Does not require a reference dataset, so compute first.
        IS_scores = get_inception_scores(args, device, num_workers)

    print("Loading Model", file=sys.stderr)
    # Get train representations
    model = load_encoder(
        args.model,
        device,
        ckpt=None,
        arch=None,
        clean_resize=args.clean_resize,
        sinception=True if args.model == "sinception" else False,
        depth=args.depth,
    )

    dataloader = get_dataloader_from_path(
        args.folder_path, model.transform, num_workers, args
    )

    representations = compute_representations(dataloader, model, device, args)



    print(f"Saving representations to {output_dir}\n", file=sys.stderr)
    save_outputs(args.output_dir, representations, args.model, None, dataloader)




if __name__ == "__main__":
    main()