from typing import Iterable
import functools
import re

import torch
from transformers import PreTrainedTokenizer
from transformers.generation_utils import GenerationMixin

@functools.lru_cache(maxsize=None)
def count_tokens(tokenizer: PreTrainedTokenizer, text: str) -> int:
    tokens = tokenizer(text, return_tensors="pt").input_ids.to("cpu")
    return tokens.shape[1]


@functools.lru_cache(maxsize=2)
def collect_stop_tokens(tokenizer: PreTrainedTokenizer) -> Iterable[int]:
    if tokenizer is None:
        return None
    vocab = tokenizer.get_vocab()
    vocab_keys = vocab.keys()

    return [vocab[key] for key in vocab_keys if re.search(r'[.!?"](\n)?$', key, re.MULTILINE) != None]


@functools.lru_cache(maxsize=2)
def collect_bad_words_ids(tokenizer: PreTrainedTokenizer, prevent_square_brackets=True, prevent_angle_brackets=True, prevent_curly_brackets=True) -> Iterable[int]:
    if tokenizer is None:
        return None
    vocab = tokenizer.get_vocab()
    vocab_keys = vocab.keys()
    bad_keys = list()

    def find_keys(char): return [
        key for key in vocab_keys if key.find(char) != -1]

    if prevent_square_brackets:
        bad_keys.extend(find_keys("["))
        # bad_keys.extend(find_keys("]"))

    if prevent_angle_brackets:
        bad_keys.extend(find_keys("<"))
        bad_keys.extend(find_keys(">"))

    if prevent_curly_brackets:
        bad_keys.extend(find_keys("{"))
        # bad_keys.extend(find_keys("}"))

    bad_words_ids = list()
    bad_keys_final = list()
    for key in bad_keys:
        if key == "</s>" or key in bad_keys_final:
            continue
        bad_id = vocab[key]
        bad_words_ids.append([bad_id])
        bad_keys_final.append(key)

    if len(bad_words_ids) < 1:
        return None
    return bad_words_ids


def generate_raw(tokenizer: PreTrainedTokenizer, model: GenerationMixin, input: str,
                 device: str = "cuda:0",
                 number_generated_tokens=60,
                 temperature=0.8,
                 tfs=None,
                 top_k=60,
                 top_p=0.9,
                 repetition_penalty=2.5,
                 repetition_penalty_range=512,
                 repetition_penalty_slope=3.33,
                 prevent_square_brackets=True,
                 prevent_angle_brackets=True,
                 prevent_curly_brackets=True):
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to("cpu")
    n_ids = input_ids.shape[1]
    if n_ids < 1:
        n_ids = 1
        input_ids = torch.tensor([[tokenizer.eos_token_id]])

    max_length = n_ids + number_generated_tokens
    torch.cuda.empty_cache()
    bad_words_ids = collect_bad_words_ids(
        tokenizer, prevent_square_brackets, prevent_angle_brackets, prevent_curly_brackets)

    gen_tokens = model.generate(
        input_ids.long().to(device),
        do_sample=True,
        min_length=max_length,
        max_length=max_length,
        temperature=temperature,
        tfs=tfs,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_penalty_range=repetition_penalty_range,
        repetition_penalty_slope=repetition_penalty_slope,
        use_cache=True,
        bad_words_ids=bad_words_ids,
        pad_token_id=tokenizer.eos_token_id
    ).long().to("cpu")[0]
    torch.cuda.empty_cache()

    stop_tokens = collect_stop_tokens(tokenizer)
    if len(gen_tokens) > n_ids:
        for i in reversed(range(len(gen_tokens))):
            if gen_tokens[i] in stop_tokens:
                gen_tokens = gen_tokens[:i+1]
                break
    output = tokenizer.decode(gen_tokens[n_ids:])
    return output
