import argparse
import json
import os
import numpy as np
import pandas as pd
import random
import torch
import time
import datasets 

def load_jsonl(file_path):
    records = []
    with open(file_path, "r") as file:
        for line in file:
            records.append(json.loads(line))
    return records

def write_jsonl(output_file, data):

    with open(output_file, "w") as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

def batch_decode_vllm(llm, prompts, batch_size=32):
    """
    Perform batch decoding using vLLM.

    Args:
    - llm: The vLLM model instance
    - prompts: List of prompts to process
    - batch_size: Number of prompts to process in each batch

    Returns:
    - List of generated responses
    """
    from vllm import SamplingParams  # type: ignore

    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        sampling_params = SamplingParams(max_tokens=llm.max_tokens, temperature=llm.temperature)
        outputs = llm.model.generate(
            batch_prompts, sampling_params
        )
        responses = [output.outputs[0].text for output in outputs]
        all_responses.extend(responses)
    return all_responses


def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_llm(args):
    if "openai" in args.model:
        from models.openai import ChatGPT
        llm = ChatGPT(model_path=args.model, max_tokens=args.max_token)
    elif "anthropic" in args.model:
        from models.anthropic import Claude
        llm = Claude(model_path=args.model, max_tokens=args.max_token)
    else:
        from models.hf import CasualLM
        llm = CasualLM(
            model_path=args.model,
            arch=args.arch,
            use_vllm=args.use_vllm,
            max_tokens=args.max_token,
            temperature=args.temperature,
        )
    return llm

def load_eval_records(args, subject):
    if args.problem_type != "clean":
        records = datasets.load_dataset('K-and-K/perturbed-knights-and-knaves',data_files=f"{args.split}/{args.problem_type}/{subject}.jsonl")["train"] 
    else:
        records = datasets.load_dataset('K-and-K/knights-and-knaves',data_files=f"{args.split}/{subject}.jsonl")["train"]
    return records

def apply_chat_template_hard_code(prompt, mode="zero"):
    if mode=="zero":
        return f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    elif mode=="normal":
        return f"""<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"""
    else:
        raise NotImplementedError
