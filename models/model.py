# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
from torch import nn
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Config,
)

from diffusers.models.normalization import RMSNorm
from diffusers import SanaTransformer2DModel, UNet2DConditionModel

from models.transformer_encoder import Qwen2Encoder

import sys
import pdb as pdb_original

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


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        cache_dir: str = "/nfshomes/asarkar6/trinity/model_weights/",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 64,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 4,
        system_prompt: str = "You will be given an image or its caption. Please describe the content of the image in detail in your own words.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.cache_dir = cache_dir
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        if "Qwen2.5-VL" in config.mllm_id:
            self.mllm_type = "qwenvl"
        elif "Qwen" in config.mllm_id:
            self.mllm_type = "qwenlm"
        elif "Llama" in config.mllm_id:
            self.mllm_type = "llamaml"
        else:
            self.mllm_type = "llavaov"

        if self.mllm_type == "llavaov":
            self.mllm_backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa", cache_dir=config.cache_dir
            )
            self.mllm_backbone.language_model.config.use_sliding_window = False
            self.mllm_backbone.language_model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.language_model.model.embed_tokens.weight.register_hook(
                freeze_hook
            )
            self.mllm_hidden_size = self.mllm_backbone.config.text_config.hidden_size
            self.mllm_backbone.language_model.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = v2.Compose([v2.Resize(384), v2.CenterCrop(384)])
            # 0.5B 896

        elif self.mllm_type == "qwenvl":
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa", cache_dir=config.cache_dir
            )
            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            self.mllm_backbone.lm_head = nn.Identity()

            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, min_pixels=min_pixels, max_pixels=max_pixels
            )
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            # 3B 2048
            # 7B 3584

        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id
        if config.num_metaqueries > 0:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<pad_token_{i}>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": ["<begin_of_img>", "<end_of_img>"]
                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<end_of_img>")

        if "Sana" in config.diffusion_model_id:
            self.transformer = SanaTransformer2DModel.from_pretrained(
                config.diffusion_model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                cache_dir=config.cache_dir
            )
            # self.transformer.caption_projection = nn.Identity()
            input_scale = math.sqrt(5.5)
            # 2304 --> 2240

        elif "stable-diffusion-v1-5" in config.diffusion_model_id:
            self.transformer = UNet2DConditionModel.from_pretrained(
                config.diffusion_model_id, subfolder="unet", cache_dir=config.cache_dir
            )
            input_scale = 1
            # 768

        else:
            raise ValueError(f"Unsupported model: {config.diffusion_model_id}")

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = (
            getattr(self.transformer.config, "caption_channels", None)
            or getattr(self.transformer.config, "encoder_hid_dim", None)
            or getattr(self.transformer.config, "cross_attention_dim", None)
        )

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.connector_in_dim,
                intermediate_size=self.connector_in_dim * 4,
                num_hidden_layers=config.connector_num_hidden_layers,
                num_attention_heads=self.connector_in_dim // 64,
                num_key_value_heads=self.connector_in_dim // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.connector = nn.Sequential(
            encoder,
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, image=None, text_response=None, add_generation_prompt=True
    ):
        if not isinstance(caption, List):
            caption = [caption]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        tokenizer.system_prompt
                        if tokenizer.mllm_type == "qwenlm"
                        else [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        else:
            suffix = (
                "\n<begin_of_img>"
                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
                + "<end_of_img><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for cap in caption
        ]

        # object images being processed here.
        if image is not None:

            # If image is not a list, wrap it in a list
            if not isinstance(image, list):
                image = [image]
            
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img in enumerate(image):
                if not isinstance(img, list):
                    image[i] = [img]

            # Resize each image in each batch if resize_fn is not None
            if tokenizer.resize_fn is not None:
                image = [
                    [tokenizer.resize_fn(sub_img) if sub_img else None for sub_img in imgs]
                    for imgs in image
                ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"} for _ in imgs]
                            + [{"type": "text", "text": cap}]
                            if all(x is not None for x in imgs)
                            else [{"type": "text", "text": cap}]
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            kwargs = {"images": [imgs for imgs in image if all(x is not None for x in imgs)]}

        elif tokenizer.mllm_type in ["qwenlm", "llamaml"]:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": cap,
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
        if tokenizer.num_metaqueries > 0:
            prompts = [p + suffix for p in prompts]
        
        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )

        if tokenizer.mllm_type == "qwenvl" and "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)
        return text_inputs.values()

    def encode_condition(
        self, input_ids, attention_mask, pixel_values, image_sizes, **kwargs
    ):
        if self.mllm_type == "llavaov":
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
            ).logits
        elif self.mllm_type == "qwenvl":
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_sizes,
                attention_mask=attention_mask,
            ).logits
        elif self.mllm_type in ["qwenlm", "llamaml"]:
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            boi_pos = torch.where(input_ids == self.boi_token_id)[1]
            eoi_pos = torch.where(input_ids == self.eoi_token_id)[1]

            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )
            mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])

            prompt_embeds = prompt_embeds[mask].view(
                batch_size, -1, prompt_embeds.size(-1)
            )
            attention_mask = attention_mask[mask].view(batch_size, -1)
        return self.connector(prompt_embeds), attention_mask

    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None):
        if isinstance(self.transformer, SanaTransformer2DModel):
            model_pred = self.transformer(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            ).sample
            return model_pred
        elif isinstance(self.transformer, UNet2DConditionModel):
            model_pred = self.transformer(
                sample=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
            ).sample
            return model_pred
        else:
            raise ValueError(
                f"Unsupported model: {self.transformer.__class__.__name__}"
            )
