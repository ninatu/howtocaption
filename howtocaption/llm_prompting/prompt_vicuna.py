import argparse
from fastchat.serve.inference import SeparatorStyle
from fastchat.train.train import smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN
from fastchat.conversation import Conversation
import pickle
import os
import tqdm
import random
import yaml
from collections import OrderedDict
from pathlib import Path

import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoModel, LlamaForCausalLM
except ImportError:
    from transformers import AutoTokenizer, AutoModelForCausalLM, LLaMATokenizer, AutoModel

from fastchat.serve.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations
from fastchat.serve.compression import compress_module


def load_model(model_path, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path,
            low_cpu_mem_usage=True, **kwargs)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        class OrderedLoader(yaml.SafeLoader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return OrderedDict(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(handle, OrderedLoader)


@torch.inference_mode()
def get_answer(model, tokenizer, asrs, config, device, debug=False):
    request_template = config['prompt']
    examples = config['examples']
    batch_processing = config.get('batch_processing', False)
    batch_size = config.get('batch_size', 6)

    messages = []
    for asr_example, answer_example in examples:
        messages.append(["Human", request_template.format(asr_example.strip())])
        messages.append(["Assistant", answer_example])

    conv_template = Conversation(
                system="A chat between a curious human and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the human's questions.",
                roles=["Human", "Assistant"],
                messages=messages,
                offset=2,
                sep_style=SeparatorStyle.SINGLE,
                sep="\n### ",
            )

    if batch_processing:
        outputs = []
        for offset in range(0, len(asrs), batch_size):

            prompts = []
            for asr in asrs[offset:offset + batch_size]:
                conv = conv_template.copy()
                conv.append_message(conv.roles[0], request_template.format(asr.strip()))
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                prompts.append(prompt)

                if debug:
                    print("Prompt:")
                    print(prompt)

            input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            output = model.generate(**input_ids, **config['model_generate_args'])
            output = tokenizer.batch_decode(output[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)
            output = [x[:-2] for x in output]
            outputs.extend(output)

            if debug:
                print("Output:")
                for x in output:
                    print(x)
    else:
        outputs = []
        for asr in asrs:
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], request_template.format(asr))
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            output = model.generate(input_ids, **config['model_generate_args'])
            output = tokenizer.decode(output[0][len(input_ids[0]):-1])
            outputs.append(output)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--asr-path", type=str)
    parser.add_argument("--model-path", type=str)

    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Use 8-bit quantization.")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    config = read_yaml(args.config)
    device = args.device
    num_gpus = args.num_gpus
    load_8bit = args.load_8bit
    debug = args.debug

    if args.debug:
        config['batch_size'] = 1

    model_path = args.model_path
    asr_path = args.asr_path
    word_blocks = config['word_blocks']
    save_dir = config['save_dir']

    exp_name = os.path.splitext(os.path.basename(args.config))[0]

    model, tokenizer = load_model(model_path, device, num_gpus, load_8bit, debug)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

    result_dir = os.path.join(save_dir, exp_name)
    os.makedirs(result_dir, exist_ok=True)

    with open(asr_path, 'rb') as fin:
        data = pickle.load(fin)

    data = list(data.items())
    if not args.debug:
        random.shuffle(data)

    for id_, (key, val) in enumerate(tqdm.tqdm(data)):
        texts = val[word_blocks]['text']

        output_path = f'{result_dir}/{key[0]}/{key[1]}/{key}.pickle'

        if os.path.exists(output_path):
            continue

        if args.debug:
            print(key, flush=True)

        results = get_answer(model, tokenizer, texts, config, device, debug=debug)

        if args.debug:
            continue

        if id_ % 100 == 0:
            print(f'Vicuna output: {results[0]}')

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as fout:
            pickle.dump(results, fout)
