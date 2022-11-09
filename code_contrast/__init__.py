import random
import termcolor
import tokenizers
import numpy as np

from copy import copy
from pathlib import Path
from itertools import groupby

from typing import List


class Encoding:
    def __init__(self, name: str):
        self.DIAMOND = 0
        self.INFILL = 0
        self.ESCAPE = 0
        self.MSG = 0
        self.FILE = 0
        self.CHUNK = 0
        self.LF = 0
        self.EOT = 0
        self._pos_tokens = []

        filename = Path(__file__).resolve().parent.parent / "encodings" / f"{name}.json"
        self._tokenizer = tokenizers.Tokenizer.from_file(str(filename))
        if name == "openai_reversible50000":
            self.EOT = 50256
            assert self.n_vocab == 50257
        elif name == "openai_programming_v1":
            self.EOT = 50256
            assert self.n_vocab == 50281
        elif name in ["openai_programming_v2", "openai_programming_v3"]:
            self.EOT = 50256
            self.LF = 198
            self.ESCAPE = self._encode_token(" §§")
            self._pos_tokens = list(range(50281, 50281 + 1024))
            assert self.decode([self._pos_tokens[0]]) == "<XXXXX>"
            assert self.decode([self._pos_tokens[-1]]) == "<VVVVV>"
            LEAVE_LESS_TPOS = 256  # TODO: this must be an arg
            # for 2000 diffs n_ctx=2048, selftest fails LEAVE_LESS_TPOS=256 => 12, LEAVE_LESS_TPOS=512 => 0
            self._pos_tokens = self._pos_tokens[:LEAVE_LESS_TPOS]
            if name == "openai_programming_v3":
                self.INFILL = 51305
                self.DIAMOND = 51306
                self.MSG = 51307
                self.FILE = 51308
                self.CHUNK = 51309
                assert self.n_vocab == 50281 + 1024 + 5
            else:
                self.INFILL = self._encode_token(" 裏覚醒")
                self.DIAMOND = self._encode_token(" ●")
                self.MSG = self._encode_token(" MSG")
                self.FILE = self._encode_token(" FILE")
                self.CHUNK = self._encode_token(" ►")
                assert self.n_vocab == 50281 + 1024
        elif name in ['fb1', 'fb3']:
            self.ESCAPE = 0
            self.DIAMOND = 1
            self.EOT = 2
            self.MSG = self._encode_token("MSG")
            self.FILE = self._encode_token("FILE")
            self.CHUNK = self._encode_token("CH")
            if name == "fb1":
                # Removed MASK tokens. Use for plain text.
                assert self.n_vocab == 50261
            else:
                # Removed MASK tokens, added position XXXXX tokens. Use to switch to diffs.
                assert self.n_vocab == 51285
                self._pos_tokens = list(range(50261, 50261 + 1024))
                assert self.decode([self._pos_tokens[0]]) == "⪦XXXXX⪧"
                assert self.decode([self._pos_tokens[-1]]) == "⪦VVVVV⪧"
        elif name == 'facebook_incoder':
            # A useless encoding for our pursoses, don't use
            self.ESCAPE = None
            self.EOT = 2
            assert self.n_vocab == 50518
        else:
            assert 0

    def _encode_token(self, text: str) -> int:
        tokens = self.encode(text)
        assert len(tokens) == 1, (text, tokens)
        return tokens[0]

    @property
    def tpos(self) -> List[int]:
        return copy(self._pos_tokens)

    @property
    def n_vocab(self) -> int:
        return self._tokenizer.get_vocab_size()

    def is_tpos(self, token: int) -> bool:
        if not self._pos_tokens:
            return False
        return self._pos_tokens[0] <= token <= self._pos_tokens[-1]

    def encode(self, sequence: str) -> List[int]:
        return self._tokenizer.encode(sequence).ids

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
        return self._tokenizer.decode(tokens)

    def hlprint(self, tokens, mask1=None, mask2=None):

        def decode_colored(tokens, mask1, mask2):
            for idx, token in enumerate(tokens):
                text = self.decode([token])
                color = None
                on_color = None
                if mask1 is not None and mask1[idx]:
                    if token == self.ESCAPE:
                        on_color = "on_green"
                    else:
                        color = "green"
                elif mask2 is not None and mask2[idx]:
                    if token == self.ESCAPE:
                        on_color = "on_magenta"
                    else:
                        color = "magenta"
                elif token == self.DIAMOND:
                    color, on_color = "red", "on_white"
                elif token in [self.ESCAPE, self.INFILL, self.MSG, self.FILE, self.CHUNK]:
                    color, on_color = "red", "on_white"
                elif token == self.EOT or self.is_tpos(token):
                    color = "red"
                yield text, color, on_color

        keyfunc = lambda text, color, on_color: (color, on_color)
        return "".join([
            termcolor.colored("".join([text for text, _, _ in group]),
                              color=color, on_color=on_color)
            for (color, on_color), group in groupby(decode_colored(tokens, mask1, mask2), keyfunc)
        ])

    # TODO: typing, unclear diffedits format
    def editclass_print(self, tokens, mask, diffedits):

        def decode_colored(tokens, mask, diffedits):
            for token, m, diffedit in zip(tokens, mask, diffedits):
                text = self.decode([token])
                color = None
                on_color = None
                if diffedit == 1:  # no edit
                    on_color = "on_blue"
                elif diffedit == 2:  # edit
                    if token == self.LF:
                        color, text = "yellow", "EDIT\n"
                    else:
                        color = "red"
                elif diffedit == 3:  # continue
                    if token == self.LF:
                        color, text = "yellow", "MOAR\n"
                    else:
                        color = "magenta"
                elif m:
                    if token == self.ESCAPE:
                        on_color = "on_green"
                    else:
                        color = "green"
                elif token == self.DIAMOND:
                    color, on_color = "grey", "on_white"
                elif token in [self.ESCAPE, self.INFILL, self.MSG, self.FILE, self.CHUNK]:
                    color, on_color = "grey", "on_white"
                else:
                    color = "blue"
                yield text, color, on_color

        keyfunc = lambda decoded, color, on_color: (color, on_color)
        return "".join([
            termcolor.colored("".join([decoded for decoded, _, _ in group]),
                              color=color, on_color=on_color)
            for (color, on_color), group in groupby(decode_colored(tokens, mask, diffedits), keyfunc)
        ])
