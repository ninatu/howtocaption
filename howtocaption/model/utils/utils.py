import os
from urllib.parse import urlparse
import torch.nn as nn

import torch
from timm.models.hub import download_cached_file

from transformers import BertTokenizer

from howtocaption.model.utils.vit import interpolate_pos_embed, VisionTransformer
from howtocaption.model.utils.vit_per_frame import VisionTransformerPerFrame

from typing import List


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'}) # there was no bos_token. After this line tokenizer.bos_token_id will appear
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)

    print('load checkpoint from %s' % url_or_filename)
    return model, msg


def create_vit_per_frame_model(image_size=384, vit='base'):
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformerPerFrame(img_size=image_size, patch_size=16, embed_dim=vision_width)
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformerPerFrame(img_size=image_size, patch_size=16, embed_dim=vision_width,
                                                   depth=24, num_heads=16)
    else:
        raise NotImplementedError
    return visual_encoder, vision_width


def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_keys:List[str]):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        print(f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized.")

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_keys: List[str],
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and all(skip_key not in module_name for skip_key in skip_keys):
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias
            if hasattr(decoder_pointer, "lora_A"):
                assert hasattr(encoder_pointer, "lora_A")
                encoder_pointer.lora_A = decoder_pointer.lora_A
            if hasattr(decoder_pointer, "lora_B"):
                assert hasattr(encoder_pointer, "lora_B")
                encoder_pointer.lora_B = decoder_pointer.lora_B
            # print(module_name+' is tied')
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_keys,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_keys)
