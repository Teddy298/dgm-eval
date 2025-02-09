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

TEST_PATHS = ['/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ReACGAN/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-WGAN-GP/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-RESFLOW/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-BigGAN-Deep/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LSGM-ODE/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-NVAE/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-MHGAN/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-LOGAN/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-ACGAN-Mod/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN-XL/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-PFGMPP/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-iDDPM-DDIM/interpolation/80%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/20%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/0%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/60%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/100%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/40%train',
              '/shared/sets/datasets/CIFAR10-dgm_eval/CIFAR10-StyleGAN2-ada/interpolation/80%train']



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




def get_relevant_paths(main_folder_path, key_value="%"):
    return [x[0] for x in os.walk(main_folder_path) if key_value in x[0]]

def create_output_paths(main_folder_path, model, paths):
    main_folder_split = main_folder_path.split('/')
    main_folder_name = main_folder_split[-1]
    reps_folder = main_folder_name + "_reps"
    outputs_paths = []
    for path in paths:
        splitted_path = path.split("/")
        pos = splitted_path.index(main_folder_name)
        subfolder_struct = splitted_path[pos+1:]
        output_path = main_folder_split[:-1]
        output_path.extend([reps_folder, model])
        output_path.extend(subfolder_struct)
        outputs_paths.append("/".join(output_path))
    return outputs_paths

def main():
    print("Starting man in get_representation")
    args = parser.parse_args()

    main_folder_path = args.folder_path
    print(f"{main_folder_path=}")
    relevant_paths = get_relevant_paths(main_folder_path)
    print(f"{relevant_paths=}")
    output_paths = create_output_paths(main_folder_path, args.model, relevant_paths)
    print(f"{output_paths=}")

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

    for img_path, output_dir in zip(relevant_paths, output_paths):
        dataloader = get_dataloader_from_path(
            img_path, model.transform, num_workers, args
        )

        representations = compute_representations(dataloader, model, device, args)

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "repr.npz")
        print(f"Saving representations to {output_path}\n", file=sys.stderr)
        hyperparams = vars(dataloader).copy()  # Remove keys that can't be pickled
        hyperparams.pop("transform")
        hyperparams.pop("data_loader")
        hyperparams.pop("data_set")

        np.savez(output_path, model=model, reps=representations, hparams=hyperparams)





if __name__ == "__main__":
    main()