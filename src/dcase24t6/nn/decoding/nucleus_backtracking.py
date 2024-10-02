#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
from typing import Any, NamedTuple, Optional, Union

import torch
from torch import Tensor, nn

from dcase24t6.nn.decoding.common import AACDecoder
import torch.nn.functional as F

pylog = logging.getLogger(__name__)


@torch.no_grad()
def generate_nucleus(
    decoder: AACDecoder,
    pad_id: int,
    bos_id: Union[int, Tensor],
    eos_id: int,
    vocab_size: int,
    frame_embs: Tensor,
    frame_embs_pad_mask: Tensor,
    max_pred_size: int = 20,
    size: int = 60,
    top_k: int = 20,
    top_p: float = 0.95,
    verbose: bool = False,
):
    frame_embs = frame_embs.permute(2, 0, 1)
    decoder.eval()
    # print(size)

    batch_finished_captions = []
    for n_sample in range(frame_embs.shape[1]):
        audio_embedding = frame_embs[:, n_sample, :].unsqueeze(1)
        repetition_counter = 0
        captions = {}
        finished_captions = []
        partial_caption = torch.tensor([[1]]).cuda()
        while (len(finished_captions) < size):
            caption_key = tuple(partial_caption.permute(1, 0)[0].tolist())
            if not caption_key in captions.keys():
                logits = decoder(
                    frame_embs=audio_embedding,
                    caps_in=partial_caption,
                    frame_embs_attn_mask=None,
                    frame_embs_pad_mask=None,
                    caps_in_attn_mask=None,
                    caps_in_pad_mask=None,
                )[-1]
                probs = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)[0]
                captions[caption_key] = probs

            next_token = torch.multinomial(captions[caption_key], num_samples=1)
            next_token = next_token.squeeze().item()

            partial_caption = torch.cat(
                (partial_caption, torch.tensor([[next_token]]).cuda()))
            if next_token == eos_id or len(partial_caption[0]) >= max_pred_size:
                finished_captions.append(partial_caption)
                if not tensor_in_list(partial_caption, finished_captions):
                    finished_captions.append(partial_caption)
                else:
                    repetition_counter += 1
                backtrack(captions, partial_caption)
                partial_caption = torch.tensor([[1]]).cuda()

        longest_caption = max(finished_captions, key=lambda t: t.size(0))
        for i in range(len(finished_captions)):
            if finished_captions[i].size(0) < longest_caption.size(0):
                padding_tensor = torch.tensor(
                    [[pad_id] * (longest_caption.size(0) - finished_captions[i].size(0))]).cuda().permute(1, 0)
                finished_captions[i] = torch.cat(
                    (finished_captions[i], padding_tensor), dim=0)
        finished_captions = torch.stack(finished_captions)[:, :, 0]
        batch_finished_captions.append(finished_captions)
    # print(f'repetitions: {repetition_counter}')
    return batch_finished_captions


def backtrack(captions, partial_caption):
    next_token = partial_caption[-1].item()
    partial_caption = partial_caption[:-1]
    decreasing_value = 1
    while partial_caption.size(0) > 0:
        caption_key = tuple(partial_caption.permute(1, 0)[0].tolist())
        decreasing_value *= captions[caption_key][next_token]

        number_of_non_zero = torch.count_nonzero(captions[caption_key])
        captions[caption_key][next_token] -= decreasing_value

        for token in captions[caption_key].nonzero():
            if token != next_token:
                captions[caption_key][token] += decreasing_value / \
                    (number_of_non_zero-1)
        next_token = partial_caption[-1].item()
        partial_caption = partial_caption[:-1]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    # pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    probs = calculate_probability(logits)
    return probs


def calculate_probability(logits):
    return F.softmax(logits, -1)


def tensor_in_list(tensor, tensor_list):
    return any(torch.equal(tensor, t) for t in tensor_list)
