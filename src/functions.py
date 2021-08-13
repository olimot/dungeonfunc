import functools
import re
from typing import Iterable, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .classes import GPTContext, GenerationOptions


def count_tokens(tokenizer: PreTrainedTokenizer, text: str) -> int:
    return memoized_count_tokens(tokenizer, text)


@functools.lru_cache(maxsize=None)
def memoized_count_tokens(tokenizer: PreTrainedTokenizer, text: str) -> int:
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
        if key == tokenizer.eos_token or key in bad_keys_final:
            continue
        bad_id = vocab[key]
        bad_words_ids.append([bad_id])
        bad_keys_final.append(key)

    if len(bad_words_ids) < 1:
        return None
    return bad_words_ids


def generate_raw(ctx: GPTContext, options: GenerationOptions, input: str):
    input_ids = ctx.tokenizer(input, return_tensors="pt").input_ids.to("cpu")
    n_ids = input_ids.shape[1]
    if n_ids < 1:
        n_ids = 1
        input_ids = torch.tensor([[ctx.tokenizer.eos_token_id]])

    max_length = min(ctx.max_position_embeddings,
                     n_ids + options.number_generated_tokens)
    print(f"Generating (input token length: {n_ids})...", end="")
    bad_words_ids = collect_bad_words_ids(
        ctx.tokenizer, options.prevent_square_brackets, options.prevent_angle_brackets, options.prevent_curly_brackets)
    torch.cuda.empty_cache()
    try:
        gen_tokens = ctx.model.generate(
            input_ids.to(ctx.device),
            do_sample=True,
            min_length=max_length,
            max_length=max_length,
            temperature=options.temperature,
            tfs=options.tfs,
            top_k=options.top_k,
            top_p=options.top_p,
            repetition_penalty=options.repetition_penalty,
            repetition_penalty_range=options.repetition_penalty_range,
            repetition_penalty_slope=options.repetition_penalty_slope,
            use_cache=True,
            bad_words_ids=bad_words_ids,
            pad_token_id=ctx.tokenizer.eos_token_id
        ).to("cpu")[0]
        output = ctx.tokenizer.decode(gen_tokens[n_ids:])
        print(f"Done (output token length: {len(gen_tokens[n_ids:])})")
        return output
    finally:
        torch.cuda.empty_cache()


def truncate_with_budget(tokenizer: PreTrainedTokenizer, text: str, budget: int, prefix="", suffix=""):
    if len(prefix):
        budget = budget - len(tokenizer.encode(prefix))
    if len(suffix):
        budget = budget - len(tokenizer.encode(suffix))

    tokens = tokenizer.encode(text)[-budget:]
    return f"{prefix}{tokenizer.decode(tokens).lstrip()}{suffix}", budget - len(tokens)


def truncate_text_list_with_budget(tokenizer: PreTrainedTokenizer, text_list: List[str], budget: int):
    text_list_truncated: List[str] = []
    for i in reversed(range(0, len(text_list))):
        text = text_list[i]
        tokens = tokenizer.encode(text)
        num_tokens = len(tokens)
        if num_tokens < budget:
            text_list_truncated = [text] + text_list_truncated
            budget = budget - num_tokens
        else:
            text_list_truncated = [tokenizer.decode(
                tokens[-budget:]).lstrip()] + text_list_truncated
            budget = 0
            break
    return text_list_truncated, budget


def build_model_input(tokenizer: PreTrainedTokenizer, max_position_embeddings: int,
                      number_generated_tokens: int = 60, memory: str = "", world_info: List[Tuple[str, str]] = [],
                      story: List[Tuple[str, str]] = [], authors_note: str = "", last_story_input: str = "",
                      section_delimiter="\n\n"):
    context_budget = int(max_position_embeddings / 2)
    story_budget = max_position_embeddings - \
        context_budget - number_generated_tokens

    last_story_input_truncated, story_budget = truncate_with_budget(tokenizer, last_story_input.strip(),
                                                                    story_budget)

    authors_note_wrapped = ""
    if len(authors_note) > 0:
        authors_note_wrapped, context_budget = truncate_with_budget(tokenizer, authors_note.strip(),
                                                                    context_budget, prefix="[Author's Note: ",
                                                                    suffix="]")

    world_info_truncated = ""
    if len(world_info) > 0:
        last_4 = story[-4:] + [last_story_input, ""]
        world_info_haystack = "".join(["".join(pair) for pair in last_4])
        world_info_found = section_delimiter.join(
            [info[1].strip() for info in world_info if world_info_haystack.find(info[0]) != -1]).strip() + section_delimiter
        world_info_truncated, context_budget = truncate_with_budget(tokenizer, world_info_found,
                                                                    context_budget)

    memory_truncated = ""
    if len(memory) > 0:
        memory_truncated, context_budget = truncate_with_budget(tokenizer, memory.strip(),
                                                                context_budget)

    story_budget = story_budget + context_budget
    list_story_text = ["".join(i).strip() + section_delimiter for i in story]
    list_story_truncated, _ = truncate_text_list_with_budget(tokenizer, list_story_text,
                                                             story_budget)
    older_story = "".join(list_story_truncated[:-4])
    last_4_story = "".join(list_story_truncated[-4:])

    return f"{memory_truncated}{world_info_truncated}{older_story}{authors_note_wrapped}{last_4_story}{last_story_input_truncated}"


def generate(ctx: GPTContext,
             options: GenerationOptions,
             memory: str = "",
             world_info: List[Tuple[str, str]] = [],
             story: List[Tuple[str, str]] = [],
             authors_note: str = "",
             last_story_input: str = "",
             section_delimiter="\n\n"):
    last_model_input = build_model_input(ctx.tokenizer, ctx.max_position_embeddings, options.number_generated_tokens, memory,
                                         world_info, story, authors_note, last_story_input, section_delimiter)
    output = generate_raw(ctx, options, last_model_input)
    new_story = story + \
        [(last_story_input, re.sub(r"\s*\n\s*\n+\s*", "\n", output))]
    return new_story
