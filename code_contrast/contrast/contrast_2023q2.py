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
# element_stretch = ["FULL", "AUX", "RAND", "EXPAND"]


@dataclass
class _PlanElement:
    el_type: str
    # el_stretch: str


@dataclass
class _FileExpandingRange:
    aux: int
    line0: int
    line1: int
    need_header: bool = True
    works0: bool = True
    works1: bool = True


@dataclass
class _File(_PlanElement):
    file_fn: str
    file_text: List[str]
    file_text_toks: List[Optional[List[int]]] = field(default_factory=lambda: [])
    formal_line0: int = -1  # §LINE1337 first line
    expanding_ranges: List[_FileExpandingRange] = field(default_factory=lambda: [])
    check_ctx_n: int = 0
    check_aux_n: int = 0
    footer_toks: List[int] = field(default_factory=lambda: [])


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

    def add_file(self, file_fn: str, file_text: List[str]):
        f = _File("FILE", file_fn, file_text)
        self.plan.append(f)
        return f

    def add_msg(self, msg_role, msg_text):
        m = _Msg("MSG", msg_role, msg_text)
        self.plan.append(m)
        return m

    # def add_chunk(self, orig_file, dest_text):
    #     c = _Chunk("CHUNK", orig_file, dest_text, [], [], -1, -1)
    #     self.plan.append(c)
    #     return c

    def from_odm_dict(
        self,
        odm: Dict[str, Any],
        limit_ctx_n: int,
        limit_aux_n: int,
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
            main_file = (fni == len(fns) - 1)
            # stretch = "RANDOM"
            # if tight_shrink:
            #     stretch = "EXPAND" if main_file else "AUX"
            f = self.add_file(fn, [(x + "\n") for x in odm["orig"][fn].splitlines()])
            if external_poi_ranges and fn in external_poi_ranges:
                line0 = max(0, min(external_poi_ranges[fn][0], len(f.file_text) - 1))
                line1 = max(0, min(external_poi_ranges[fn][1], len(f.file_text) - 1))
                f.expanding_ranges.append(_FileExpandingRange(
                    aux=(1 if (tight_shrink and not main_file) else 0),
                    line0=line0,
                    line1=line1,
                ))
            files.append(f)
        self.add_msg("USER", odm["commitmsg"])
        for fn, f in zip(fns, files):
            chunks.extend(self.run_diff(f, [(x + "\n") for x in odm["dest"][fn].splitlines()], exact_cx_lines0, exact_cx_lines1))
        random.shuffle(chunks)
        self.plan.extend(chunks)
        self.pack_context(2, limit_ctx_n, limit_aux_n)
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
            c =  _Chunk("CHUNK", f, dest_text, i0, i1, j0, j1)
            chunks.append(c)
            f.expanding_ranges.append(_FileExpandingRange(
                    aux=0,
                    line0=i0,
                    line1=i1-1,
                ))
        return chunks

    def assign_random_line0_to_files(self):
        MINLINE = 1000
        MAXLINE = 9000
        for f in self.plan:
            if f.el_type == "FILE":
                f.formal_line0 = random.randint(MINLINE, MAXLINE)

    def pack_context(self, mask_from_plan_n: int, limit_ctx_n: int, limit_aux_n: int):
        filled_ctx_n = 0
        filled_aux_n = 0
        self.assign_random_line0_to_files()
        LINE_NUMBER_EACH = 15   # lines
        self.r, self.m = [], []
        plan_toks: List[List[int]] = [list() for _ in range(len(self.plan))]
        plan_mask: List[List[int]] = [list() for _ in range(len(self.plan))]

        def dump_MSG(i, msg: _Msg):
            toks = self.enc.encode(msg.msg_role + " " + msg.msg_text)
            plan_toks[i].extend(toks)
            plan_mask[i].extend([1]*len(toks))

        def _file_line2toks_helper(file: _File, er: _FileExpandingRange, l: int, aux: bool):
            print("line2toks", l, len(file.file_text))
            nonlocal filled_ctx_n, filled_aux_n
            if l < 0 or l >= len(file.file_text):
                return False
            if file.file_text_toks[l] is not None:
                return False
            t = self.enc.encode(file.file_text[l])
            if aux:
                if filled_aux_n + len(t) < limit_aux_n:
                    filled_aux_n += len(t)
                    file.check_aux_n += len(t)
                    file.file_text_toks[l] = t
                    return True
            else:
                if filled_ctx_n + len(t) < limit_ctx_n + (limit_aux_n - filled_aux_n):
                    filled_ctx_n += len(t)
                    file.check_ctx_n += len(t)
                    file.file_text_toks[l] = t
                    return True
            return False

        toks_count_typical_header = len([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % 1234))

        def init_FILE(i, file: _File):
            t = self.enc.encode("FILE " + file.file_fn.replace("\n", "\\n") + "\n")
            plan_toks[i].extend(t)
            plan_mask[i].extend([1]*len(t))
            file.footer_toks = [self.enc.ESCAPE] + self.enc.encode("/FILE\n")
            file.check_ctx_n = len(t) + len(file.footer_toks)
            file.check_aux_n = 0
            # Each range has a header, until it bumps into another range above when exanding
            file.file_text_toks = [None] * len(file.file_text)
            for er in file.expanding_ranges:
                er.need_header = True
                for line in range(er.line0, er.line1 + 1):
                    _file_line2toks_helper(file, er, line, aux=er.aux)

        def expand_FILE(i, file: _File) -> bool:
            nonlocal filled_ctx_n, filled_aux_n
            anything_works = False
            for ri, er in enumerate(file.expanding_ranges):
                print("range%d: %d..%d, %d, %d, aux=%d" % (i, er.line0, er.line1, er.works0, er.works1, er.aux))
                if er.works0:
                    success = _file_line2toks_helper(file, er, er.line0, aux=er.aux)
                    if not success and er.line0 > 0:
                        # We bumped into another expanding range
                        er.ranges_need_header = False
                        if er.aux:
                            filled_aux_n -= toks_count_typical_header
                        else:
                            filled_ctx_n -= toks_count_typical_header
                    if success and er.line0 == 0:
                        er.works0 = False
                    elif success:
                        er.line0 -= 1
                    else:
                        er.works0 = False
                if er.works1:
                    success = _file_line2toks_helper(file, er, er.line1 + 1, aux=er.aux)  # For example we start with the range (5, 5) and expand from there, the line below is 6
                    if success and er.line1 + 1 >= len(file.file_text) - 1:
                        er.works1 = False
                        er.line1 = len(file.file_text) - 1
                    elif success:
                        er.line1 += 1
                        assert er.line1 < len(file.file_text), ri
                    else:
                        er.works1 = False
                anything_works |= er.works0 or er.works1
            return anything_works

        def finish_FILE(i, file: _File):
            t, m = [], []
            assert len(file.file_text) == len(file.file_text_toks)
            for er in file.expanding_ranges:
                assert er.line1 < len(file.file_text), file.expanding_ranges
                assert er.works0 == False and er.works1 == False
                line_countdown = 0
                for line in range(er.line0, er.line1):
                    if line_countdown == 0:
                        line_t = [self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % (er.line0 + file.formal_line0))
                        t.extend(line_t)
                        m.extend([1 if line > er.line0 else 0]*len(line_t))
                        line_countdown = 15
                    line_t = file.file_text_toks[line]
                    assert line_t is not None, line
                    t.extend(line_t)
                    m.extend([1]*len(line_t))
            plan_toks[i].extend(t)
            plan_mask[i].extend(m)
            plan_toks[i].extend(file.footer_toks)
            plan_mask[i].extend([1]*len(file.footer_toks))

        def dump_CHUNK(i, chunk: _Chunk):
            t = self.enc.encode("CHUNK\n")
            for line in range(chunk.i0, chunk.i1):
                line_t = self.enc.encode(chunk.orig_file.file_text[line])
                t.extend(line_t)
            t.extend([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % chunk.formal_line))
            for j in range(chunk.j0, chunk.j1):
                t.extend(self.enc.encode(chunk.dest_text[j]))
            m = [1]*len(t)
            plan_toks[i] = t
            plan_mask[i] = m

        switch_init = {"MSG": dump_MSG, "FILE": init_FILE, "CHUNK": dump_CHUNK}
        switch_expand = {"FILE": expand_FILE}
        switch_finish = {"FILE": finish_FILE}
        for i, p in enumerate(self.plan):
            switch_init[p.el_type](i, p)
            filled_ctx_n += len(plan_toks[i])
        print("after init, filled_ctx_n %d" % (filled_ctx_n,))
        while 1:
            any_still_expanding = False
            for i, p in enumerate(self.plan):
                if p.el_type not in switch_expand:
                    continue
                any_still_expanding |= switch_expand[p.el_type](i, p)
            if not any_still_expanding:
                break
        for i, p in enumerate(self.plan):
            if p.el_type not in switch_finish:
                continue
            switch_finish[p.el_type](i, p)
        for i, (lst, msk) in enumerate(zip(plan_toks, plan_mask)):
            self.r.append(self.enc.ESCAPE)
            self.m.append(1 if i >= mask_from_plan_n else 0)
            self.r.extend(lst)
            self.m.extend(msk if i >= mask_from_plan_n else [0]*len(msk))
        self.r.extend([self.enc.ESCAPE, self.enc.EOT])
        self.m.extend([1, 1])
        assert len(self.r) == len(self.m)
        print("filled_ctx_n %d < limit %d" % (filled_ctx_n, limit_ctx_n))
        print("filled_aux_n %d < limit %d" % (filled_aux_n, limit_aux_n))
        print("filled_ctx_n + filled_aux_n = %d" % (filled_ctx_n + filled_aux_n))

    def dump_r(self):
        return hlprint(enc, self.r, self.m)

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



def self_test(enc: SMCEncoding, odm: Dict[str, Any], verbose: bool, limit_ctx_n=2048, limit_aux_n=512, tight_shrink: bool=False):
    import time
    t0 = time.time()
    test1 = Contrast2023q2(enc)
    full_orig_tokens = test1.from_odm_dict(odm, limit_ctx_n, limit_aux_n,
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
    self_test(enc, example_odm, verbose=True, limit_ctx_n=512, limit_aux_n=128)

