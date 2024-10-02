#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
from typing import Any, NamedTuple, Optional, Union

import torch
from torch import Tensor, nn
from dcase24t6.nn.decoding.common import AACDecoder
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

pylog = logging.getLogger(__name__)


class MCTSNode:
    audio_embedding = None
    top_k_childs = 5
    top_p = 0.5

    def __init__(self, partial_caption, parent=None, lprob=0):
        self.partial_caption = partial_caption
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.is_terminal = False
        self.lprob = lprob
        self.probabilities = None

    def ucb1(self, exploration_weight=20):
        if self.visits == 0:
            return float('inf')
        print(self.value / self.visits, math.sqrt(math.log(self.parent.visits) / self.visits))
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def puct(self, exploration_weight=10):
        if self.visits == 0:
            return float('inf')
        return self.value + exploration_weight * self.parent.probabilities[self.partial_caption[-1]] * math.sqrt(self.parent.visits) / (1 + self.visits)

    def add_child(self, token, lprob):
        child = MCTSNode(torch.cat((self.partial_caption,
                                    torch.tensor([[token]]).cuda())), parent=self, lprob=lprob+self.lprob)
        self.children[token] = child
        return child


def mcts_select(root):
    node = root
    while node.children:
        if not all(child.visits > 0 for child in node.children.values()):
            return node
        non_terminal_children = [
            child for child in node.children.values() if not child.is_terminal]
        if not non_terminal_children:
            return node
        else:
            node = max(non_terminal_children, key=lambda child: child.puct())
    return node


def mcts_expand(node, decoder, max_pred_size, eos_id):
    if node.is_terminal or (len(node.children) > 0 and all(child.is_terminal for child in node.children.values())):
        return node

    if not node.children:
        if node.probabilities is None:
            logits = decoder(
                frame_embs=MCTSNode.audio_embedding,
                caps_in=node.partial_caption,
                frame_embs_attn_mask=None,
                frame_embs_pad_mask=None,
                caps_in_attn_mask=None,
                caps_in_pad_mask=None,
            )[-1]
            node.probabilities = top_k_top_p_filtering(
                logits, top_k=MCTSNode.top_k_childs, top_p=MCTSNode.top_p)[0]
        non_zero_indices = torch.nonzero(node.probabilities).squeeze()
        # check if non_zero_indices is a single element tensor
        if len(non_zero_indices.shape) == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        for token in non_zero_indices:
            if token not in node.children:
                child = node.add_child(
                    token, lprob=torch.log(node.probabilities[token]))
                if token == eos_id or len(child.partial_caption) == max_pred_size:
                    child.is_terminal = True

    for child in node.children.values():
        if child.visits == 0:
            return child


def mcts_simulate(node, decoder, max_length, eos_id):
    if node.is_terminal:
        return node.partial_caption, node.lprob
    # if all childs are terminal
    if len(node.children) > 0 and all(child.is_terminal for child in node.children.values()):
        return node.partial_caption, node.lprob
    current_caption = node.partial_caption.clone()
    log_probs = node.lprob

    while len(current_caption) < max_length:
        logits = decoder(
            frame_embs=MCTSNode.audio_embedding,
            caps_in=current_caption,
            frame_embs_attn_mask=None,
            frame_embs_pad_mask=None,
            caps_in_attn_mask=None,
            caps_in_pad_mask=None,
        )[-1]

        probabilities = top_k_top_p_filtering(
            logits, top_k=MCTSNode.top_k_childs, top_p=MCTSNode.top_p)[0]
        token = torch.multinomial(probabilities, 1).item()
        current_caption = torch.cat((current_caption, torch.tensor([[token]]).cuda()))
        log_probs += torch.log(probabilities[token])
        if token == eos_id:
            break
    return current_caption, log_probs


def mcts_backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value = ((node.visits - 1) * node.value + value) / node.visits
        node = node.parent


def select_best_caption(root, decoder, max_length, eos_id, verbose=False):
    node = root
    while node.children:
        best_child = max(node.children.values(), key=lambda child: child.visits)
        node = best_child
        # if node.is_terminal:
        if len(node.children) == 0:
            break
    if best_child.partial_caption[-1] != eos_id:
        if verbose:
            print('Warning: best caption does not end with EOS token')
        return (mcts_simulate(best_child, decoder, max_length, eos_id)[0].permute(1, 0),
                best_child.partial_caption.permute(1, 0),
                best_child.lprob)
    return (best_child.partial_caption.permute(1, 0), best_child.partial_caption.permute(1, 0), best_child.lprob)


def evaluate_caption(caption, lprob, caption_embedding_model, tokenizer, audio_tokenizer):
    string_caption = audio_tokenizer.decode(caption.squeeze().tolist())
    batch_dict = tokenizer(string_caption, padding=True,
                           truncation=True, return_tensors='pt')
    for key in batch_dict:
        batch_dict[key] = batch_dict[key].cuda()

    with torch.no_grad():
        outputs = caption_embedding_model(**batch_dict)

    caption_embedding = mean_pooling(outputs, batch_dict['attention_mask'])
    caption_embedding = caption_embedding[:, :256]
    mean_audio_embedding = MCTSNode.audio_embedding.mean(0)
    embeddings = torch.cat([mean_audio_embedding, caption_embedding], dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    score = ((embeddings[:1] @ embeddings[1:].T) * 100).squeeze()

    return score


def pick_node_for_next_iteration(node):
    if node.is_terminal:
        return node
    return max(node.children.values(), key=lambda child: child.visits)


@ torch.no_grad()
def generate_mcts(
    decoder: AACDecoder,
    pad_id: int,
    bos_id: Union[int, Tensor],
    eos_id: int,
    vocab_size: int,
    audio_tokenizer: AACTokenizer,
    frame_embs: Tensor,
    frame_embs_pad_mask: Tensor,
    max_pred_size: int = 20,
    max_iterations: int = 1000,
    max_iterations_per_sample: int = 100,
    top_k_childs: int = 3,
    top_p: float = 0.9,
    verbose: bool = False,
):
    frame_embs = frame_embs.permute(2, 0, 1)
    MCTSNode.audio_embedding = frame_embs
    MCTSNode.top_k_childs = top_k_childs
    MCTSNode.top_p = top_p

    model_path = 'nomic-ai/nomic-embed-text-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    caption_embedding_model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True).cuda()

    decoder.eval()
    caption_embedding_model.eval()

    best_captions = []
    for n_sample in tqdm(range(frame_embs.shape[1])):
        MCTSNode.audio_embedding = frame_embs[:, n_sample, :].unsqueeze(1)
        root = MCTSNode(partial_caption=torch.full((1, 1), bos_id).cuda(), parent=None)
        while not root.is_terminal:
            root = MCTSNode(partial_caption=root.partial_caption, parent=None)
            for _ in range(max_iterations_per_sample):
                node = mcts_select(root)
                if not node.is_terminal:
                    if len(root.children) == 1:
                        # print('only one child')
                        break
                    child = mcts_expand(node, decoder, max_pred_size, eos_id)
                    caption, lprob = mcts_simulate(
                        child, decoder, max_pred_size, eos_id)
                    value = evaluate_caption(caption, lprob,
                                             caption_embedding_model, tokenizer, audio_tokenizer)
                    mcts_backpropagate(child, value)
            root = pick_node_for_next_iteration(root)

        best_caption = select_best_caption(root, decoder, max_pred_size, eos_id, verbose)
        best_caption = root.partial_caption.permute(1, 0)
        best_captions.append(best_caption)

    longest_caption = max(best_captions, key=lambda x: x.shape[1])
    # Pad all captions to the same length
    for i, caption in enumerate(best_captions):
        padded_caption = caption
        if caption.shape[1] < longest_caption.shape[1]:
            padded_caption = torch.cat([caption, torch.full(
                (1, longest_caption.shape[1] - caption.shape[1]), pad_id).cuda()], dim=1)
        best_captions[i] = padded_caption
    best_captions = torch.cat(best_captions, dim=0)
    return {'predictions': best_captions}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return F.softmax(logits, -1)
