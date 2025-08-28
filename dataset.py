# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, Image
import PIL
import io
import os
import torch
import glob
from torchvision.transforms import v2
import random
from torch.utils.data.dataset import ConcatDataset
from functools import partial
import numpy as np

import sys
import pdb as pdb_original

# global variables
BACKUP_PATH = os.path.join("/nfshomes/asarkar6/aditya/PRISM/", "backup")

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


def delete_keys_except(batch, except_keys):
    keys_to_delete = [key for key in list(batch.keys()) if key not in except_keys]
    for key in keys_to_delete:
        del batch[key]
    return batch

def create_object_images(bbox_info, img_mat):
    obj_img_lst = []
    for idx, item in enumerate(bbox_info):
        x_min = int(item["xmin"])
        x_max = int(item["xmax"])
        y_min = int(item["ymin"])
        y_max = int(item["ymax"])

        if (x_max-x_min)*(y_max-y_min)>0 and y_max>y_min and x_max>x_min:
            obj_img = np.asarray(img_mat)[y_min:y_max, x_min:x_max]
            if obj_img.shape[0]==0 or obj_img.shape[1]==0 or obj_img.shape[2]!=3:
                raise ValueError("The object image size is invalid.")
            else:
                obj_img_lst.append(PIL.Image.fromarray(obj_img).convert("RGB"))
        else:
            raise ValueError("The object image size is invalid.")

    return obj_img_lst

def create_dummy_objects(num_objs=3):
    caption = "A smiling woman with a pink umbrella stands in front of a sign, with a serene lake and green trees in the background."
    target_img = PIL.Image.open(os.path.join(BACKUP_PATH, "temp_img.jpg")).convert("RGB")

    obj_img_lst = []
    for idx in range(num_objs):
        obj_img_lst.append(PIL.Image.open(os.path.join(BACKUP_PATH, "temp_obj_"+str(idx)+".jpg")).convert("RGB"))

    return caption, target_img, obj_img_lst


def _i2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_image_mask = rand_probs <= 0.1
    images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def i2i_eval_process_fn(batch):
    images = batch["image"]
    captions = ["" for _ in range(len(images))]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch


def _t2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption for caption in captions]
    for i in range(len(images)):
        try:
            images[i] = PIL.Image.open(
                io.BytesIO(images[i]["bytes"])
                if images[i]["bytes"] is not None
                else images[i]["path"]
            ).convert("RGB")
        except:
            images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_caption_mask = rand_probs < 0.1
    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    batch["caption"] = captions
    delete_keys_except(batch, ["target", "caption"])
    return batch

def t2i_eval_process_fn(batch):
    captions = batch["caption"]
    batch["caption"] = captions
    delete_keys_except(batch, ["caption"])
    return batch


def _inst_process_fn(batch, target_transform):
    source_images = batch["source_images"]
    caption = batch["caption"]
    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = [
        caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)
    ]
    source_images = (
        [
            (
                image
                if not null_image_mask[i]
                else [PIL.Image.new("RGB", (img.width, img.height)) for img in image]
            )
            for i, image in enumerate(source_images)
        ]
        if source_images is not None
        else None
    )
    batch["caption"], batch["input_images"] = caption, source_images
    batch["target"] = [
        target_transform(img.convert("RGB")) for img in batch["target_image"]
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch


def inst_eval_process_fn(batch):
    source_images = batch["source_images"]
    caption = batch["caption"]
    batch["caption"], batch["input_images"] = caption, source_images
    delete_keys_except(batch, ["caption", "input_images"])
    return batch


def _editing_process_fn(batch, target_transform, ground_truth_transform):
    source_images = batch["source_image"]
    target_images = batch["target_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    for i in range(len(source_images)):
        try:
            source_images[i] = PIL.Image.open(
                io.BytesIO(source_images[i]["bytes"])
                if source_images[i]["bytes"] is not None
                else source_images[i]["path"]
            ).convert("RGB")
            target_images[i] = PIL.Image.open(
                io.BytesIO(target_images[i]["bytes"])
                if target_images[i]["bytes"] is not None
                else target_images[i]["path"]
            ).convert("RGB")
        except:
            source_images[i] = None
            target_images[i] = None
            captions[i] = ""

    batch["target"] = [
        target_transform(image) if image is not None else None
        for image in target_images
    ]
    rand_probs = torch.rand((len(target_images), 1))
    null_image_mask = rand_probs <= 0.1
    source_images = [
        (
            PIL.Image.new("RGB", (image.width, image.height))
            if (image is not None and null_image_mask[i])
            else image
        )
        for i, image in enumerate(source_images)
    ]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch

def _trinity_process_fn(batch, target_transform, stage=2):
    target_images = batch["target_image"]
    object_images = batch["object_image"]
    captions = batch["caption"]

    # Validate that all lists have the same length
    if not (len(target_images) == len(object_images) == len(captions)):
        min_length = min(len(target_images), len(object_images), len(captions))
        print(f"Warning: Mismatched lengths detected: target_images={len(target_images)}, object_images={len(object_images)}, captions={len(captions)}. Truncating to {min_length}.")
        target_images = target_images[:min_length]
        object_images = object_images[:min_length]
        captions = captions[:min_length]

    batch_size = len(target_images)
    transformed_objects = [None] * batch_size  # Store transformed images

    for i in range(batch_size):
        flag = 0
        try:
            target_images[i] = PIL.Image.open(
                io.BytesIO(target_images[i]["bytes"])
                if target_images[i]["bytes"] is not None
                else target_images[i]["path"]
            ).convert("RGB")
            if stage == 2:
                if len(object_images[i]) > 0:
                    try:
                        object_images[i] = create_object_images(object_images[i], target_images[i])
                        # Test target_transform and store result
                        transformed_objects[i] = {idx: target_transform(image) if image is not None else None for idx, image in enumerate(object_images[i])}
                    except:
                        print(f"Target transform failed for object_images[{i}].")
                        flag = 1
                else:
                    flag = 1
            else:
                flag = 0
        except:
            print(f"Going into except block for image loading or object creation at index {i}.")
            flag = 1
        
        if flag == 1:
            captions[i], target_images[i], object_images[i] = create_dummy_objects()
            transformed_objects[i] = None  # Reset transformed objects for dummy data

    batch["target"] = [target_transform(image) if image is not None else None for image in target_images]
    batch["caption"] = [item for item in captions]

    if stage == 2:
        try:
            batch["input_images"] = [
                transformed_objects[i] if transformed_objects[i] is not None else
                {idx: target_transform(image) if image is not None else None for idx, image in enumerate(item)}
                for i, item in enumerate(object_images)
            ]
        except:
            print("Going into except block for input_images transform.")
            for i in range(batch_size):
                if transformed_objects[i] is None and any(image is None or not isinstance(image, PIL.Image.Image) for image in object_images[i]):
                    captions[i], target_images[i], object_images[i] = create_dummy_objects()
                    transformed_objects[i] = None
            batch["target"] = [target_transform(image) if image is not None else None for image in target_images]
            batch["caption"] = [item for item in captions]
            batch["input_images"] = [{idx: image for idx, image in enumerate(item)} for item in object_images]

        delete_keys_except(batch, ["target", "input_images", "caption"])
    else:
        delete_keys_except(batch, ["target", "caption"])
    return batch
    # returns batch["target", "input_images", "caption"]
    
def editing_eval_process_fn(batch):
    source_images = batch["source_image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption[-1] for caption in captions]
    batch["caption"], batch["input_images"] = captions, [
        [image] if image is not None else None for image in source_images
    ]
    delete_keys_except(batch, ["input_images", "caption"])
    return batch

def trinity_eval_process_fn(batch, stage=2):
    captions = batch["caption"]
    target_images = batch["target_image"]
    object_images = batch["object_image"] if stage == 2 else [None]*len(target_images)

    if stage == 2:
        for i in range(len(object_images)):
            flag = 0
            if len(object_images[i]) > 0:
                try:
                    object_images[i] = create_object_images(object_images[i], target_images[i])
                except:
                    print("Going into except block.")
                    flag = 1
            else:
                flag = 1
            
            if flag == 1:
                captions[i], target_images[i], object_images[i] = create_dummy_objects()
        
        batch["input_images"] = [image if image is not None else None for item in object_images for image in item]
        batch["caption"]= captions
        delete_keys_except(batch, ["input_images", "caption"])
    else:
        batch["caption"] = captions
        delete_keys_except(batch, ["caption"])

    return batch

def _collate_fn(batch, tokenize_func, tokenizer, stage=2):
    none_idx = [i for i, example in enumerate(batch) if example["target"] is None]
    if len(none_idx) > 0:
        batch = [example for i, example in enumerate(batch) if i not in none_idx]
    return_dict = {"target": torch.stack([example["target"] for example in batch])}

    # input_images will be a list of list of object images
    if stage == 2:
        input_images = [
            list(example["input_images"].values()) if "input_images" in example and example["input_images"][0] is not None else None
            for example in batch
        ]

    captions = [example["caption"] for example in batch]

    if stage == 2:
        if len(input_images) > 0 and all(x is not None for x in input_images):
            (
                return_dict["input_ids"],
                return_dict["attention_mask"],
                return_dict["pixel_values"],
                return_dict["image_sizes"],
            ) = tokenize_func(
                tokenizer, captions, input_images
            )
        else:
            return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
                tokenizer, captions
            )
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
                tokenizer, captions
            )
        
    return return_dict
    # returns target, input_ids, attention_mask, pixel_values, image_sizes


def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):
    train_datasets = {}
    if "cc12m_i2i" in data_args.train_datasets:
        train_dataset = load_dataset(
            "pixparse/cc12m-wds",
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )

        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(10000))

        if (
            data_args.train_datasets["cc12m_i2i"] > 0
            and training_args.run_name != "test"
        ):
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(
                range(int(data_args.train_datasets["cc12m_i2i"] * 1000000))
            )
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if not col in (["image", "caption"])
            ]
        )
        train_datasets["cc12m_i2i"] = train_dataset

    if "cc12m_t2i" in data_args.train_datasets:
        train_dataset = load_dataset(
            "pixparse/cc12m-wds",
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )

        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(10000))

        if (
            data_args.train_datasets["cc12m_t2i"] > 0
            and training_args.run_name != "test"
        ):
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(
                range(int(data_args.train_datasets["cc12m_t2i"] * 1000000))
            )

        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if not col in (["image", "caption"])
            ]
        )
        train_datasets["cc12m_t2i"] = train_dataset

    if "inst2m" in data_args.train_datasets:
        train_dataset = load_dataset(
            "xcpan/MetaQuery_Instruct_2.4M_512res",
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(10000))

        if data_args.train_datasets["inst2m"] > 0 and training_args.run_name != "test":
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(
                range(int(data_args.train_datasets["inst2m"] * 1000000))
            )
        train_dataset = train_dataset.rename_column("prompt", "caption")
        train_datasets["inst2m"] = train_dataset

    if "ominiedit" in data_args.train_datasets:
        train_dataset = load_dataset(
            "TIGER-Lab/OmniEdit-Filtered-1.2M",
            cache_dir=training_args.data_dir,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )
        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(10000))

        if (
            data_args.train_datasets["ominiedit"] > 0
            and training_args.run_name != "test"
        ):
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(
                range(int(data_args.train_datasets["ominiedit"] * 1000000))
            )

        train_dataset = train_dataset.rename_column("src_img", "source_image")
        train_dataset = train_dataset.rename_column("edited_img", "target_image")
        train_dataset = train_dataset.rename_column("edited_prompt_list", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if not col in (["source_image", "target_image", "caption"])
            ]
        )
        train_datasets["ominiedit"] = train_dataset

    if "trinity" in data_args.train_datasets:
        # iterate over all files
        # paths = [os.path.join(training_args.data_dir, path) for path in glob.glob(os.path.join(training_args.data_dir, "*.jsonl"))]
        path = os.path.join(training_args.data_dir, "trinity-data-real-combined.json")
        train_dataset = load_dataset(
            "json",
            data_files = path,
            split="train",
            num_proc=training_args.datasets_num_proc,
        )

        if training_args.run_name == "test":
            train_dataset = train_dataset.select(range(20))

        if (
            data_args.train_datasets["trinity"] > 0
            and training_args.run_name != "test"
        ):
            train_dataset = train_dataset.shuffle(seed=training_args.data_seed)
            train_dataset = train_dataset.select(
                range(int(data_args.train_datasets["trinity"] * 1000000))
            )

        train_dataset = train_dataset.rename_column("file_name", "target_image")
        train_dataset = train_dataset.rename_column("object", "object_image")
        train_dataset = train_dataset.rename_column("prompt", "caption")

        # add the prefix if its not there.
        train_dataset = train_dataset.map(lambda example: {"target_image": os.path.join(training_args.data_dir, example["target_image"])})

        train_datasets["trinity"] = train_dataset

    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
            # v2.Normalize([0.5], [0.5]),
        ]
    )

    ground_truth_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )

    i2i_process_fn = partial(_i2i_process_fn, target_transform=target_transform)
    t2i_process_fn = partial(_t2i_process_fn, target_transform=target_transform)
    inst_process_fn = partial(_inst_process_fn, target_transform=target_transform)
    editing_process_fn = partial(
        _editing_process_fn,
        target_transform=target_transform,
        ground_truth_transform=ground_truth_transform,
    )
    if training_args.training_stage == 2:
        trinity_process_fn = partial(_trinity_process_fn, target_transform=target_transform)
        collate_fn = partial(_collate_fn, tokenize_func=tokenize_func, tokenizer=tokenizer)
        trinity_eval_process_fn_full = partial(trinity_eval_process_fn, stage=2)
    else:
        trinity_process_fn = partial(_trinity_process_fn, target_transform=target_transform, stage=1)
        collate_fn = partial(_collate_fn, tokenize_func=tokenize_func, tokenizer=tokenizer, stage=1)
        trinity_eval_process_fn_full = partial(trinity_eval_process_fn, stage=1)

    eval_dataset = train_datasets[data_args.eval_dataset].select(
        range(training_args.world_size)
    )

    if "source_images" in eval_dataset.column_names:
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))
    elif "source_image" in eval_dataset.column_names:
        eval_dataset = eval_dataset.cast_column("source_image", Image(decode=True))
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))
    elif "object_image" in eval_dataset.column_names:
        eval_dataset = eval_dataset.cast_column("target_image", Image(decode=True))
    else:
        eval_dataset = eval_dataset.cast_column("image", Image(decode=True))
    
    gt_images = (
        eval_dataset["target_image"]
        if "target_image" in eval_dataset.column_names
        else eval_dataset["image"]
    )
    gt_images = [ground_truth_transform(image.convert("RGB")) for image in gt_images]

    if data_args.eval_dataset in ["cc12m_i2i"]:
        eval_dataset.set_transform(i2i_eval_process_fn)
    elif data_args.eval_dataset in ["cc12m_t2i"]:
        eval_dataset.set_transform(t2i_eval_process_fn)
    elif data_args.eval_dataset in ["inst2m"]:
        eval_dataset.set_transform(inst_eval_process_fn)
    elif data_args.eval_dataset in ["ominiedit"]:
        eval_dataset.set_transform(editing_eval_process_fn)
    elif data_args.eval_dataset in ["trinity"]:
        eval_dataset.set_transform(trinity_eval_process_fn_full)
    else:
        raise ValueError(f"Unknown eval_dataset: {data_args.eval_dataset}")

    for dataset_name, train_dataset in train_datasets.items():
        if dataset_name in ["cc12m_i2i"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(i2i_process_fn)
        elif dataset_name in ["cc12m_t2i"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(t2i_process_fn)
        elif dataset_name in ["inst2m"]:
            train_datasets[dataset_name].set_transform(inst_process_fn)
        elif dataset_name in ["ominiedit"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "source_image", Image(decode=False)
            )
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "target_image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(editing_process_fn)
        elif dataset_name in ["trinity"]:
            train_datasets[dataset_name] = train_datasets[dataset_name].cast_column(
                "target_image", Image(decode=False)
            )
            train_datasets[dataset_name].set_transform(trinity_process_fn)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # shuffle the dataset
        train_datasets[dataset_name] = train_datasets[dataset_name].shuffle(
            seed=training_args.data_seed
        )

    # if more than one dataset in the dict, concatenate them
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(list(train_datasets.values()))
    else:
        train_dataset = train_datasets[list(train_datasets.keys())[0]]

    return train_dataset, eval_dataset, gt_images, collate_fn
