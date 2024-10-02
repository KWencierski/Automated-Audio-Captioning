#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, ClassVar, Sequence, TypeGuard

from tokenizers import Encoding, Regex, Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.normalizers import Lowercase, Normalizer, Replace
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.normalizers import Strip, StripAccents
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import PostProcessor
from tokenizers.processors import Sequence as ProcessorSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import Trainer, WordLevelTrainer, BpeTrainer

from dcase24t6.tokenization.aac_tokenizer import AACTokenizer


class PretrainedTokenizer(AACTokenizer):

    VERSION: ClassVar[int] = 1

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        version: int | None = None,
    ) -> None:
        super().__init__(tokenizer, pad_token, bos_token, eos_token, unk_token, version)

    @classmethod
    def default_tokenizer(
        cls,
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
    ) -> Tokenizer:
        raise NotImplementedError

    def token_to_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def get_token_to_id(self, with_added_tokens: bool = True) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def get_id_to_token(self, with_added_tokens: bool = True) -> dict[int, str]:
        return {
            id_: token for token, id_ in self.get_token_to_id(with_added_tokens).items()
        }

    def train_from_iterator(
        self,
        sequence: list[str],
        trainer: Trainer | None = None,
    ) -> None:
        pass

    def encode(self, sequence: str, disable_unk_token: bool = False) -> Encoding:
        unk_token = self.tokenizer.unk_token
        if disable_unk_token:
            self.tokenizer.unk_token = ""
        encoding = self.tokenizer.encode(sequence)
        if disable_unk_token:
            self.tokenizer.unk_token = unk_token
        return encoding

    def encode_batch(
        self,
        sequence: list[str],
        disable_unk_token: bool = False,
    ) -> list[Encoding]:
        unk_token = self.tokenizer.unk_token
        if disable_unk_token:
            self.tokenizer.unk_token = ""
        encodings = self.tokenizer.batch_encode_plus(sequence, padding=True)
        if disable_unk_token:
            self.tokenizer.unk_token = unk_token
        return encodings

    def decode(
        self,
        sequence: Sequence[int] | Encoding,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(sequence, Encoding):
            sequence = sequence.ids

        decoded = self.tokenizer.decode(
            sequence, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def decode_batch(
        self,
        sequences: Sequence[Sequence[int]] | Sequence[Encoding],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if is_list_encoding(sequences):
            sequences = [element.ids for element in sequences]
        decoded = self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )
        return decoded

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def is_list_encoding(sequence: Any) -> TypeGuard[Sequence[Encoding]]:
    return isinstance(sequence, Sequence) and all(
        isinstance(element, Encoding) for element in sequence
    )
