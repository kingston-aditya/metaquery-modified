# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from argparse import ArgumentParser
from pipeline_metaquery import MetaQueryPipeline
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from trainer_utils import find_newest_checkpoint

from torchvision.transforms import v2

import pdb as pdb_original
import sys

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

target_transform = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.CenterCrop((224, 224)),
        ]
    )

def sample_metaquery(pipeline, prompt, image, args):
    return pipeline(
        caption=prompt,
        image=image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        enable_progress_bar=False,
    ).images

def create_objects(args, object_images):
    temp_list = []
    for objs in object_images:
        a = [target_transform(Image.open(os.path.join(args.dataset_folder, item["img_pth"]))) for item in objs]
        temp_list.append(a)
    return temp_list


def main(args):
    # load the metaquery pipeline
    args.checkpoint_path = find_newest_checkpoint(args.checkpoint_path)
    print("Checkpoint", args.checkpoint_path)
    pipeline = MetaQueryPipeline.from_pretrained(
        args.checkpoint_path,
        ignore_mismatched_sizes=True,
        _gradient_checkpointing=False,
        torch_dtype=torch.bfloat16,
    )
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    sample_fn = sample_metaquery
    os.makedirs(args.output_dir, exist_ok=True)

    # load the dataset
    dataset = load_dataset(
        "json",
        split="train",
        data_files=os.path.join(args.dataset_folder, "metadata.jsonl"),
        num_proc=1,
    )
    dataset = dataset.select(
        range(args.start_idx, args.end_idx if args.end_idx != -1 else len(dataset))
    )

    dataset = dataset.rename_column("prompt", "caption")
    dataset = dataset.rename_column("object", "object_images")

    # do the inference
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        data = dataset[i : i + args.batch_size]
        prompt = data["caption"]

        if args.inference_type == 2:
            obj_images = create_objects(args, data["object_images"])
            images = sample_fn(pipeline, prompt, obj_images, args)
        else:
            images = sample_fn(pipeline, prompt, None, args)

        for j, image in enumerate(images):
            image.save(f"{args.output_dir}/{args.start_idx+i+j:05d}.png")

        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default="/path/to/dataset_folder")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    parser.add_argument("--output_dir", type=str, default="/path/to/output_dir")
    parser.add_argument("--checkpoint_path", type=str, default="/path/to/checkpoint_path")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    # 1 for prompt only inference, 2 for object images + prompt inference
    parser.add_argument("--inference_type", type=int, default=2)
    args = parser.parse_args()
    main(args)
