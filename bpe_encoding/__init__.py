import os, json, random
import tokenizers
import numpy as np
from typing import Optional, Any, List


class Encoding:
    def __init__(self, name):
        self.is_tpos = lambda i: False
        self.DIAMOND = 0
        self.INFILL = 0
        self.ESCAPE = 0
        self.MSG = 0
        self.FILE = 0
        self.CHUNK = 0
        if name == "openai_reversible50000":
            self.json_fn = os.path.join(os.path.dirname(__file__), "openai_reversible50000.json")
            self.izer = tokenizers.Tokenizer.from_file(self.json_fn)
            self.EOT = 50256
            assert self.izer.get_vocab_size() == 50257
        elif name == "openai_programming_v1":
            self.json_fn = os.path.join(os.path.dirname(__file__), "openai_programming_v1.json")
            self.izer = tokenizers.Tokenizer.from_file(self.json_fn)
            self.EOT = 50256
            assert self.izer.get_vocab_size() == 50281
        elif name in ["openai_programming_v2", "openai_programming_v3"]:
            self.json_fn = os.path.join(os.path.dirname(__file__), "%s.json" % name)
            self.izer = tokenizers.Tokenizer.from_file(self.json_fn)
            self.EOT = 50256
            self.tpos = list(range(50281, 50281 + 1024))
            self.is_tpos = lambda i: (50281 <= i < 51305)
            t = self.izer.decode([self.tpos[0], self.tpos[-1]])
            assert t == "<XXXXX><VVVVV>"
            tmp = self.encode(" §§")
            assert len(tmp) == 1
            self.ESCAPE = tmp[0]
            self.LF = 198
            LEAVE_LESS_TPOS = 256
            # for 2000 diffs n_ctx=2048, selftest fails LEAVE_LESS_TPOS=256 => 12, LEAVE_LESS_TPOS=512 => 0
            self.tpos = self.tpos[:LEAVE_LESS_TPOS]
            if name == "openai_programming_v3":
                self.INFILL = 51305
                self.DIAMOND = 51306
                self.MSG = 51307
                self.FILE = 51308
                self.CHUNK = 51309
                assert self.izer.get_vocab_size() == 50281 + 1024 + 5
            else:
                self.INFILL = 25992    # " 裏覚醒"
                self.DIAMOND = 48049   # " ●"
                t1 = self.encode(" MSG")
                t2 = self.encode(" FILE")
                assert len(t1) == 1 and len(t2) == 1, (t1, t2)
                self.MSG = t1[0]
                self.FILE = t2[0]
                self.CHUNK = 34933     # " ►"
                assert self.izer.get_vocab_size() == 50281 + 1024
        elif name == 'facebook_incoder':
            # A useless encoding for our pursoses, don't use
            self.izer = tokenizers.Tokenizer.from_file(
                os.path.join(os.path.dirname(__file__), "facebook_incoder.json"),
                )
            self.ESCAPE = None
            self.EOT = 2
            assert self.izer.get_vocab_size() == 50518
        elif name in ['fb1', 'fb3']:
            self.izer = tokenizers.Tokenizer.from_file(
                os.path.join(os.path.dirname(__file__), f"{name}.json"),
                )
            if name == "fb1":
                # Removed MASK tokens. Use for plain text.
                assert self.izer.get_vocab_size() == 50261
            else:
                # Removed MASK tokens, added position XXXXX tokens. Use to switch to diffs.
                assert self.izer.get_vocab_size() == 51285
                self.tpos = list(range(50261, 50261 + 1024))
                self.is_tpos = lambda i: (50261 <= i < 51285)
                t = self.izer.decode([self.tpos[0], self.tpos[-1]])
                assert "XXXXX" in t
                assert "VVVVV" in t
            self.ESCAPE = 0
            self.DIAMOND = 1
            self.EOT = 2
            self.CHUNK = 3
            t1 = self.encode("MSG")
            t2 = self.encode("FILE")
            t3 = self.encode("CH")
            assert len(t1) == 1 and len(t2) == 1 and len(t3) == 1, (t1, t2, t3)
            self.MSG = t1[0]
            self.FILE = t2[0]
            self.CHUNK = t3[0]
        else:
            assert 0
        self.senc: Optional[Any] = None
        self.n_vocab = self.izer.get_vocab_size()

    def encode(self, s):
        t = self.izer.encode(s)
        return t.ids

    def decode(self, lst, skip_zeros: bool=False, cut_at_eot: bool=False):
        if isinstance(lst, np.ndarray):
            assert len(lst.shape) == 1, lst.shape
            x = lst
        else:
            x = np.array(lst)
        if skip_zeros:
            i = np.argmax(x > 0)
            x = x[i:]
        if cut_at_eot:
            i = np.argmax(x == self.EOT)
            if i > 0:
                x = x[:i]
        t = self.izer.decode(x)
        return t

    def hlprint(self, lst, mask1=None, mask2=None):
        import termcolor
        r = ""
        i = 0
        current_color = (None, None)
        currect_accum = ""
        def add_with_color(color, on_color, s):
            nonlocal r, current_color, i, currect_accum
            if current_color != (color, on_color):
                if len(currect_accum):
                    r += termcolor.colored(currect_accum, color=current_color[0], on_color=current_color[1])
                currect_accum = ""
                current_color = (color, on_color)
            currect_accum += s
        while 1:
            if i >= len(lst):
                break
            t = lst[i]
            if mask1 is not None and mask1[i]:
                if t == self.ESCAPE:
                    add_with_color(None, "on_green", self.izer.decode([t]))
                else:
                    add_with_color("green", None, self.izer.decode([t]))
            elif mask2 is not None and mask2[i]:
                if t == self.ESCAPE:
                    add_with_color(None, "on_magenta", self.izer.decode([t]))
                else:
                    add_with_color("magenta", None, self.izer.decode([t]))
            elif t == self.DIAMOND:
                add_with_color("red", "on_white", self.izer.decode([t]))
            elif t in [self.ESCAPE, self.INFILL, self.MSG, self.FILE, self.CHUNK]:
                add_with_color("red", "on_white", self.izer.decode([t]))
            elif t == self.EOT or self.is_tpos(t):
                add_with_color("red", None, self.izer.decode([t]))
            else:
                add_with_color(None, None, self.izer.decode([t]))
            i += 1
        add_with_color("finish", "finish", "")
        return r

    def editclass_print(self, lst, mask, diffedits):
        import termcolor
        r = ""
        i = 0
        current_color = (None, None)
        currect_accum = ""
        def add_with_color(color, on_color, s):
            nonlocal r, current_color, i, currect_accum
            if current_color != (color, on_color):
                if len(currect_accum):
                    r += termcolor.colored(currect_accum, color=current_color[0], on_color=current_color[1])
                currect_accum = ""
                current_color = (color, on_color)
            currect_accum += s
        while 1:
            if i >= len(lst):
                break
            t = lst[i]
            if diffedits[i]==1:  # no edit
                add_with_color(None, "on_blue", self.izer.decode([t]))
            elif diffedits[i]==2:  # edit
                if t==self.LF:
                    add_with_color("yellow", None, "EDIT\n")
                else:
                    add_with_color("red", None, self.izer.decode([t]))
            elif diffedits[i]==3:  # continue
                if t==self.LF:
                    add_with_color("yellow", None, "MOAR\n")
                else:
                    add_with_color("magenta", None, self.izer.decode([t]))
            elif mask[i]:
                if t == self.ESCAPE:
                    add_with_color(None, "on_green", self.izer.decode([t]))
                else:
                    add_with_color("green", None, self.izer.decode([t]))
            elif t == self.DIAMOND:
                add_with_color("grey", "on_white", self.izer.decode([t]))
            elif t in [self.ESCAPE, self.INFILL, self.MSG, self.FILE, self.CHUNK]:
                add_with_color("grey", "on_white", self.izer.decode([t]))
            else:
                add_with_color("blue", None, self.izer.decode([t]))
            i += 1
        add_with_color("finish", "finish", "")
        return r

    def load_cpp(self):
        if self.senc is not None:
            return
        import pyximport
        pyximport.install()
        from bpe_encoding import cpp_encoder
        j = json.load(open(self.json_fn))
        vocab = j["model"]["vocab"]
        merges = j["model"]["merges"]
        vocab2 = dict()
        for v, i in vocab.items():
            vocab2[v.encode("utf-8")] = i
        merges2 = list()
        for m in merges:
            merges2.append(m.encode("utf-8"))
        self.senc = cpp_encoder.StochEncoder()
        self.senc.save_vocab(vocab2)
        self.senc.save_merges(merges2)

    def encode_stochastic(self, s, bounds_at: List[int], prob: float):
        bounds_n = int(len(s) * prob)
        if len(bounds_at) > 0:
            assert bounds_at[0] == 0
            assert bounds_at[-1] == len(s)
            bounds_at = list(set(bounds_at))
            bounds_at.sort()
        else:
            bounds_set = set([random.randint(0, len(s) - 1) for _ in range(bounds_n)])
            bounds_set.add(len(s))
            bounds_set.add(0)
            bounds_at = list(bounds_set)
            bounds_at.sort()
        # print("bounds_at", bounds_at, " len(bounds_at)", len(bounds_at), " len(s)", len(s))
        if len(bounds_at) == 1:  # set() eats equal elements, bad for zero-length strings
            bounds_at = [0, len(s)]
        result = []
        for i in range(len(bounds_at)-1):
            result.extend(self.encode(s[bounds_at[i] : bounds_at[i+1]]))
        return result, bounds_at
