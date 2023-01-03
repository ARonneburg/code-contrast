import random
import numpy as np

from copy import copy
from pathlib import Path
from typing import List, Tuple

import tiktoken
from tiktoken.load import load_tiktoken_bpe


__all__ = ["SMCEncoding"]


class SMCEncoding:
    def __init__(self, name: str):
        self.DIAMOND = 0
        self.INFILL = 0
        self.ESCAPE = 0
        self.MSG = 0
        self.FILE = 0
        self.CHUNK = 0
        self.LF = 0
        self.LFLF = 0
        self.EOT = 0
        self._pos_tokens = []
        self._tokenizer = None

        if name in ["openai_reversible50000"]:
            self.EOT = 50256
            pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            special_tokens = {
                "<|endoftext|>": 50256,
            }
            mergeable_ranks = load_tiktoken_bpe("az://openaipublic/encodings/r50k_base.tiktoken")
            self._tik = tiktoken.Encoding(
                name,
                pat_str=pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens,
            )
            self.n_vocab = self._tik.n_vocab
            assert self.n_vocab == 50257
            self.LF = self._encode_token("\n")
            assert self.LF == 198
            self.LFLF = self._encode_token("\n\n")

        elif name in ["openai_cl100k", "openai_programming_v2"]:
            if name == "openai_cl100k":
                tt_name = "az://openaipublic/encodings/cl100k_base.tiktoken"
                pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
                special_tokens = {
                    "<|endoftext|>": 100257,
                    "<|fim_prefix|>": 100258,
                    "<|fim_middle|>": 100259,
                    "<|fim_suffix|>": 100260,
                    "<|endofprompt|>": 100276,
                    "►": 100277,
                    "●": 100278,
                    "§": 100279,
                }
                self.EOT = 100257
                self.INFILL = 100259
                self.CHUNK = 100277
                self.DIAMOND = 100278
                self.ESCAPE = 100279
                last_special_plus_one = 100280
            elif name == "openai_programming_v2":
                pat_str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                tt_name = "az://openaipublic/encodings/p50k_base.tiktoken"
                special_tokens = {
                    "<|endoftext|>": 50256,
                    # "  ": 50257
                    # ...space tokens...
                    # "                         ": 50280,
                }
                self.EOT = 50256
                self.ESCAPE = 47171   # " §§"
                self.INFILL = 25992   # " 裏覚醒"
                self.DIAMOND = 48049  # " ●"
                self.CHUNK = 34933    # " ►"
                last_special_plus_one = 50281
            else:
                assert 0
            chars = "XYZV"
            position_tokens = ["⪦" +
                    chars[i//4//4//4//4 % 4] +
                    chars[i//4//4//4 % 4] +
                    chars[i//4//4 % 4] +
                    chars[i//4 % 4] +
                    chars[i % 4] + "⪧"
                    for i in range(1024)]
            for i, postok in enumerate(position_tokens):
                special_tokens[postok] = last_special_plus_one + i
            self._pos_tokens = list(range(last_special_plus_one, last_special_plus_one + 1024))

            mergeable_ranks = load_tiktoken_bpe(tt_name)
            self._tik = tiktoken.Encoding(
                name,
                pat_str=pat_str,
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens,
            )
            self.n_vocab = self._tik.n_vocab
            if name == "openai_cl100k":
                assert self.n_vocab == 101304
            elif name == "openai_programming_v2":
                assert self.n_vocab == 51305
            else:
                assert 0
            self.MSG = self._encode_token(" MSG")
            self.FILE = self._encode_token(" FILE")
            self.LF = self._encode_token("\n")
            assert self.LF == 198
            self.LFLF = self._encode_token("\n\n")
            LEAVE_LESS_TPOS = 256
            self._pos_tokens = self._pos_tokens[:LEAVE_LESS_TPOS]
            # for i in range(self._tik.n_vocab):
            #     print("%05i \"%s\"" % (i, self._tik.decode([i]).replace("\n", "\\n")))

        elif name in ['fb1', 'fb3']:
            import tokenizers
            filename = Path(__file__).resolve().parent / f"{name}.json"
            self._tokenizer = tokenizers.Tokenizer.from_file(str(filename))
            self.ESCAPE = 0
            self.DIAMOND = 1
            self.EOT = 2
            self.MSG = self._encode_token("MSG")
            self.FILE = self._encode_token("FILE")
            self.CHUNK = self._encode_token("CH")
            self.n_vocab = self._tokenizer.get_vocab_size()
            if name == "fb1":
                # Removed MASK tokens. Use for plain text.
                assert self.n_vocab == 50261
            else:
                # Removed MASK tokens, added position XXXXX tokens.
                assert self.n_vocab == 51285
                self._pos_tokens = list(range(50261, 50261 + 1024))
                assert self.decode([self._pos_tokens[0]]) == "⪦XXXXX⪧"
                assert self.decode([self._pos_tokens[-1]]) == "⪦VVVVV⪧"
        else:
            assert 0

    def _encode_token(self, text: str) -> int:
        if self._tokenizer:
            tokens = self._tokenizer.encode(text).ids
        else:
            tokens = self._tik.encode_ordinary(text)
        assert len(tokens) == 1, (text, tokens)
        return tokens[0]

    @property
    def tpos(self) -> List[int]:
        return copy(self._pos_tokens)

    def is_tpos(self, token: int) -> bool:
        if not self._pos_tokens:
            return False
        return self._pos_tokens[0] <= token <= self._pos_tokens[-1]

    def encode(self, txt: str) -> List[int]:
        if self._tokenizer:
            return self._tokenizer.encode(txt).ids
        else:
            result = []
            cursor = 0
            while 1:
                slash_n = txt.find("\n", cursor)
                if slash_n == -1:
                    more = self._tik.encode_ordinary(txt[cursor:])
                    result.extend(more)
                    break
                else:
                    more = self._tik.encode_ordinary(txt[cursor:slash_n])
                    result.extend(more)
                    result.append(self.LF)
                cursor = slash_n + 1
            return result

    def encode_stochastic(self, sequence, bounds_at: List[int], prob: float) -> Tuple[List[int], List[int]]:
        bounds_n = int(len(sequence) * prob)
        if len(bounds_at) > 0:
            assert bounds_at[0] == 0
            assert bounds_at[-1] == len(sequence)
            bounds_at = list(set(bounds_at))
            bounds_at.sort()
        else:
            bounds_set = set([random.randint(0, len(sequence) - 1)
                              for _ in range(bounds_n)])
            bounds_set.add(len(sequence))
            bounds_set.add(0)
            bounds_at = list(bounds_set)
            bounds_at.sort()
        if len(bounds_at) == 1:  # set() eats equal elements, bad for zero-length strings
            bounds_at = [0, len(sequence)]
        result = []
        for a, b in zip(bounds_at[:-1], bounds_at[1:]):
            result.extend(self.encode(sequence[a:b]))
        return result, bounds_at

    def decode(self, tokens, skip_zeros: bool = False, cut_at_eot: bool = False) -> str:
        if isinstance(tokens, np.ndarray):
            assert len(tokens.shape) == 1, tokens.shape
        else:
            tokens = np.array(tokens)
        if skip_zeros:
            i = np.argmax(tokens > 0)
            tokens = tokens[i:]
        if cut_at_eot:
            i = np.argmax(tokens == self.EOT)
            if i > 0:
                tokens = tokens[:i]
        if self._tokenizer:
            return self._tokenizer.decode(tokens)
        else:
            return self._tik.decode(tokens)

