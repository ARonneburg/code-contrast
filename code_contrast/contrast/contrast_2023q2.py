import random
import copy
import json
import re

import termcolor

import difflib
from cdifflib import CSequenceMatcher

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.contrast.contrast_stochastic import ops_remove_short_equals
from code_contrast.contrast.contrast_stochastic import ops_stochastic_expand
from code_contrast.print_utils import editclass_print, hlprint

from collections import defaultdict
from dataclasses import dataclass, field

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


element_types = ["FILE", "USER", "ASSISTANT", "CHUNK", "TOOL", "OUTPUT"]
element_stretch = ["FULL", "AUX", "RAND", "EXPAND"]


@dataclass
class _PlanElement:
    el_type: str
    el_stretch: str


@dataclass
class _File(_PlanElement):
    file_fn: str
    file_text: List[str]
    file_ranges: List[Tuple[int, int]] = field(default_factory=lambda: [])     # line numbers
    file_line0: int = -1    # context will have line0..line1 present, indexes in file_text[]
    file_line1: int = -1
    formal_line0: int = -1  # §LINE1337 first line


@dataclass
class _Chunk(_PlanElement):
    orig_file: _File
    dest_text: List[str]
    i0: int
    i1: int
    j0: int
    j1: int
    formal_line: int = -1
    shift: int = -1
    # for decode only
    fuzzy: int = -1
    error: str = ""


@dataclass
class _Msg(_PlanElement):
    msg_role: str
    msg_text: str


class Contrast2023q2:
    def __init__(self, enc: SMCEncoding):
        self.enc: SMCEncoding = enc
        self.r: List[int] = list()
        self.m: List[int] = list()
        self.plan: List[_PlanElement] = list()
        self.cx_lines0 = 2

    def add_file(self, stretch: str, file_fn: str, file_text: List[str]):
        f = _File("FILE", stretch, file_fn, file_text)
        self.plan.append(f)
        return f

    def add_msg(self, msg_role, msg_text):
        m = _Msg("MSG", "FULL", msg_role, msg_text)
        self.plan.append(m)
        return m

    # def add_chunk(self, orig_file, dest_text):
    #     c = _Chunk("CHUNK", "FULL", orig_file, dest_text, [], [], -1, -1)
    #     self.plan.append(c)
    #     return c

    def from_odm_dict(
        self,
        odm: Dict[str, Any],
        n_ctx: int,
        # random_shrink = True,
        tight_shrink = False,
        exact_cx_lines0 = -1,
        exact_cx_lines1 = -1,
        external_poi_ranges: Optional[DefaultDict[str, List[Tuple[int, int]]]] = None,
    ):
        files1 = list(odm["orig"].keys())
        files2 = list(odm["dest"].keys())
        assert files1 == files2
        fns = list(files1)
        if tight_shrink:
            fns.reverse()   # main file moves to the end, more visible to the model
        else:
            random.shuffle(fns)
        files = []
        chunks = []
        for fni, fn in enumerate(fns):
            stretch = "RANDOM"
            if tight_shrink:
                stretch = "AUX" if (fni != len(fns) - 1) else "EXPAND"
            f = self.add_file(stretch, fn, [(x + "\n") for x in odm["orig"][fn].splitlines()])
            if external_poi_ranges and fn in external_poi_ranges:
                f.file_ranges.extend(external_poi_ranges[fn])
            files.append(f)
        self.add_msg("USER", odm["commitmsg"])
        for fn, f in zip(fns, files):
            chunks.extend(self.run_diff(f, [(x + "\n") for x in odm["dest"][fn].splitlines()], exact_cx_lines0, exact_cx_lines1))
        random.shuffle(chunks)
        self.plan.extend(chunks)
        self.pack_context(n_ctx)
        import IPython; IPython.embed(); quit()

    def run_diff(self, f: _File, dest_text: List[str], exact_cx_lines0: int, exact_cx_lines1: int):
        # an important side effect of this function is f.file_ranges
        chunks = []
        if len(f.file_text)==0:
            f.file_text.append("\n")
        if f.file_text[-1][-1] != "\n":
            f.file_text[-1] += "\n"
        if dest_text[-1][-1] != "\n":
            dest_text[-1] += "\n"
        lines_diff = list(CSequenceMatcher(None, f.file_text, dest_text).get_opcodes())
        lines_diff = ops_stochastic_expand(lines_diff,
            left_prob=1, right_prob=1,
            exact_cx_lines0=exact_cx_lines0, exact_cx_lines1=exact_cx_lines1,
            disable_insert=True   # we don't like pure inserts, because without deleted lines the position to delete is only defined by the line number, therefore model arithmetic
        )
        lines_diff = ops_remove_short_equals(lines_diff, upto=2)
        for op, i0, i1, j0, j1 in lines_diff:
            if op == "equal":
                continue
            assert op in ["replace", "joined", "insert"], op
            c =  _Chunk("CHUNK", "FULL", f, dest_text, i0, i1, j0, j1, -1, -1)
            chunks.append(c)
        return chunks

    def pack_context(self, n_ctx):
        self.r, self.m = [], []
        plan_tokens: List[List[int]] = [list() for _ in range(len(self.plan))]
        def dump_MSG(i, msg: _Msg):
            plan_tokens[i] = [self.enc.ESCAPE] + self.enc.encode(msg.msg_role + " " + msg.msg_text) + [self.enc.DIAMOND]
        def dump_FILE(i, file: _File):
            t = [self.enc.ESCAPE] + self.enc.encode("FILE " + file.file_fn.replace("\n", "\\n") + "\n")
            line_countdown = 0
            for l in range(len(file.file_text)):
                if line_countdown == 0:
                    t.extend([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % (l + file.formal_line0)))
                    line_countdown = 15
                t.extend(self.enc.encode(file.file_text[l]))
                line_countdown -= 1
            t.append(self.enc.DIAMOND)
            plan_tokens[i] = t
        def dump_CHUNK(i, chunk: _Chunk):
            t = [self.enc.ESCAPE] + self.enc.encode("CHUNK\n")
            for line in range(chunk.i0, chunk.i1):
                t.extend(self.enc.encode(chunk.orig_file.file_text[line]))
            t.extend([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % chunk.formal_line))
            for j in range(chunk.j0, chunk.j1):
                t.extend(self.enc.encode(chunk.dest_text[j]))
            t.append(self.enc.DIAMOND)
            plan_tokens[i] = t
        el_switch = {"MSG": dump_MSG, "FILE": dump_FILE, "CHUNK": dump_CHUNK}
        for i, p in enumerate(self.plan):
            el_switch[p.el_type](i, p)
        # join list of lists into one lists
        for lst in plan_tokens:
            self.r.extend(lst)
            self.m.extend([1]*len(lst))

    def dump_r(self):
        return hlprint(enc, self.r)

    def __repr__(self):
        ret = ""
        import termcolor
        x: _PlanElement
        for x in self.plan:
            ret += termcolor.colored(x.el_type, "white", attrs=["bold"]) + " "
            for field in x.__dataclass_fields__:
                if field == "el_type":
                    continue
                ret += field + " "
                val = repr(getattr(x, field))
                if len(val) > 40:
                    val = val[:40] + "... "
                else:
                    val = val + " "
                ret += termcolor.colored(val, "cyan") + " "
            ret += "\n"
        return ret


test_orig = """
from typing import Callable
import math

def newton_method(f: Callable[[float], float], x1: float, x2: float) -> float:

    asertr x1 < x2, "x1 must be less than x2"
    while x2 - x1 > 1e-6:
        x = (x1 + x2) / 2
        if f(x) == 0:
            return x
        elif f(x) * f(x1) < 0:
            x2 = x
        else:
            x1 = x
    x /= 0
    return x


print("This form of precession is specific to Einstein's theory of general relativity. These results confirm its existence in the most extreme physical event we can observe, the collision of two black holes.")
print("In the fastest example previously measured from orbiting neutron stars called binary pulsars, it took over 75 years for the orbit to precess")

if __name__ == "__main__":
    print(newton_method(lambda x: x ** 2 - 1, 0, 10-1))
"""

test_dest = """
from typing import Callable
import math

def newton_method(f: Callable[[float], float], x1: float, x2: float) -> float:
    assert x1 < x2, "x1 must be less than x2"
    while x2 - x1 > 1e-6:
        x = (x1 + x2) / 2
        if f(x) == 0:
            return x
        elif f(x) * f(x1) < 0:
            x2 = x
        else:
            x1 = x
    return x


# print("This form of precession is specific to Einstein's theory of general relativity. These results confirm its existence in the most extreme physical event we can observe, the collision of two black holes.")
# print("In the fastest example previously measured from orbiting neutron stars called binary pulsars, it took over 75 years for the orbit to precess")

if __name__ == "__main__":
    print(newton_method(lambda x: x ** 2 - 1, 0, 10-1))
    print("Better!")
"""


example_odm = {
    "orig": {
        'file1.py': test_orig,
    },
    "dest": {
        'file1.py': test_dest,
    },
    "commitmsg": "fix typo",
}



def self_test(enc: SMCEncoding, odm: Dict[str, Any], verbose: bool, n_ctx: int, tight_shrink: bool=False):
    import time
    t0 = time.time()
    test1 = Contrast2023q2(enc)
    full_orig_tokens = test1.from_odm_dict(odm, n_ctx,
        tight_shrink=tight_shrink,
    )
    quit()

    test1.write_edits()
    if verbose:
        t1 = time.time()
        print("prompt %0.2fms => %i tokens" % (1000*(t1 - t0), len(test1.r)))
    if len(test1.r) > 2*n_ctx:
        # Don't test because likely there will not be enough position tokens anyway
        return {}
    edit_classes = test1.edit_class_vector()
    if verbose:
        print(editclass_print(enc, test1.r, test1.m, edit_classes))
        print("tokens %i, n_ctx=%i" % (len(test1.r), n_ctx))
    test2 = ContrastDiff(enc)
    test_odm_nodest = copy.deepcopy(odm)
    del test_odm_nodest["dest"]
    us = test2.untokenize(test1.r, full_orig_tokens)
    e1 = test1.dump_edits()
    e2 = test2.dump_edits()
    if verbose:
        print("\n" + termcolor.colored("-"*130, "yellow"))
        print(e1)
    def first_different_symbol_e1_e2():
        for i in range(len(e1)):
            if e1[i] != e2[i]:
                return i
        return -1
    assert e1 == e2, ("len(test1.r)==%i" % len(test1.r)) + "\n" + e1 + "\n" + e2 + "\n\nfirst_different_symbol_e1_e2=%i" % first_different_symbol_e1_e2()
    test2.apply_edits_return_dest(us)
    for err in test2.errors:
        print("ERROR:", err)
    for fn in test1.dest_tokens.keys():
        # if verbose:
        #     print("dest %s:" % fn)
        #     print(hlprint(enc, test1.dest_tokens[fn]))
        if test1.dest_tokens[fn] != test2.dest_tokens[fn]:
            dest1 = enc.decode(test1.dest_tokens[fn])
            dest2 = enc.decode(test2.dest_tokens[fn])
            udiff = list(difflib.unified_diff(
                dest1.splitlines(),
                dest2.splitlines(),
                fromfile=fn,
                tofile=fn,
                lineterm="",
            ))
            print("\n".join(udiff))
            print(json.dumps(us.stats))
            assert 0, len(udiff)
    if verbose:
        print(json.dumps(us.stats))
        print("diff.r", len(test1.r))
    return us.stats


if __name__ == "__main__":
    enc = SMCEncoding("openai_cl100k")
    self_test(enc, example_odm, verbose=True, n_ctx=512)

