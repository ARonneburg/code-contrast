import random
import time
import copy
import json

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


element_types = ["SYSTEM", "USER", "ASSISTANT", "FILE", "CHUNK", "TOOL", "OUTPUT"]
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
    check_ctx_n: int = 0
    check_aux_n: int = 0


@dataclass
class _File(_PlanElement):
    file_fn: str
    file_lines: List[str]
    file_lines_toks: List[Optional[List[int]]] = field(default_factory=lambda: [])
    formal_line0: int = -1  # §LINE1337 first line
    expanding_ranges: List[_FileExpandingRange] = field(default_factory=lambda: [])
    check_lines: int = 0
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

    def add_file(self, file_fn: str, file_lines: List[str]):
        f = _File("FILE", file_fn, file_lines)
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
            # main_file = (fni == len(fns) - 1)
            # stretch = "RANDOM"
            # if tight_shrink:
            #     stretch = "EXPAND" if main_file else "AUX"
            f = self.add_file(fn, [(x + "\n") for x in odm["orig"][fn].splitlines()])
            if external_poi_ranges and fn in external_poi_ranges:
                poi_list = external_poi_ranges[fn]
                for line0, line1 in poi_list:
                    f.expanding_ranges.append(_FileExpandingRange(
                        aux=1,
                        line0=max(0, min(line0, len(f.file_lines) - 1)),
                        line1=max(0, min(line1, len(f.file_lines) - 1)),
                    ))
            files.append(f)
        self.add_msg("USER", odm["commitmsg"])
        for fn, f in zip(fns, files):
            chunks.extend(self.run_diff(f, [(x + "\n") for x in odm["dest"][fn].splitlines()], exact_cx_lines0, exact_cx_lines1))
        random.shuffle(chunks)
        self.plan.extend(chunks)
        self.pack_context(1, limit_ctx_n, limit_aux_n)

    def run_diff(self, f: _File, dest_text: List[str], exact_cx_lines0: int, exact_cx_lines1: int):
        # an important side effect of this function is f.expanding_ranges
        chunks = []
        if len(f.file_lines)==0:
            f.file_lines.append("\n")
        if f.file_lines[-1][-1] != "\n":
            f.file_lines[-1] += "\n"
        if dest_text[-1][-1] != "\n":
            dest_text[-1] += "\n"
        lines_diff = list(CSequenceMatcher(None, f.file_lines, dest_text).get_opcodes())
        lines_diff = ops_stochastic_expand(lines_diff,
            left_prob=1, right_prob=1,
            exact_cx_lines0=exact_cx_lines0, exact_cx_lines1=exact_cx_lines1,
            disable_insert=True   # we don't like pure inserts, because without deleted lines the position to delete is only defined by the line number, therefore model arithmetic
        )
        lines_diff = ops_remove_short_equals(lines_diff, upto=2)
        for op, i0, i1, j0, j1 in lines_diff:
            if op == "equal":
                continue
            assert op in ["replace", "joined", "insert", "delete"], op
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

    def pack_context(self, mask_from_plan_n: int, limit_ctx_n: int, limit_aux_n: int) -> Tuple[int, int]:
        self.assign_random_line0_to_files()
        LINE_NUMBER_EACH = 15   # lines
        self.r, self.m = [], []
        plan_toks: List[List[int]] = [list() for _ in range(len(self.plan))]
        plan_mask: List[List[int]] = [list() for _ in range(len(self.plan))]

        def dump_MSG(i, msg: _Msg):
            toks = self.enc.encode(msg.msg_role + " " + msg.msg_text + "\n")
            plan_toks[i].extend(toks)
            plan_mask[i].extend([1]*len(toks))

        def dump_CHUNK(i, chunk: _Chunk):
            t = self.enc.encode("CHUNK\n")
            for line in range(chunk.i0, chunk.i1):
                line_t = self.enc.encode(chunk.orig_file.file_lines[line])
                t.extend(line_t)
            t.extend([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % chunk.formal_line))
            for j in range(chunk.j0, chunk.j1):
                t.extend(self.enc.encode(chunk.dest_text[j]))
            m = [1]*len(t)
            plan_toks[i] = t
            plan_mask[i] = m

        toks_count_LINE = len([self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % 1234))

        def init_FILE(i, file: _File):
            nonlocal filled_ctx_n, filled_aux_n
            t = self.enc.encode("FILE " + file.file_fn.replace("\n", "\\n") + "\n")
            plan_toks[i].extend(t)
            plan_mask[i].extend([1]*len(t))
            file.footer_toks = [self.enc.ESCAPE] + self.enc.encode("/FILE\n")
            filled_aux_n += len(t) + len(file.footer_toks)
            file.check_lines = 0
            # Each range has a header, until it bumps into another range above when exanding
            file.file_lines_toks = [None] * len(file.file_lines)
            for er in file.expanding_ranges:
                er.need_header = True
                if er.aux:
                    filled_aux_n += toks_count_LINE
                else:
                    filled_ctx_n += toks_count_LINE
                for line in range(er.line0, er.line1 + 1):
                    _file_line2toks_helper(file, er, line, aux=er.aux)

        def _file_line2toks_helper(file: _File, er: _FileExpandingRange, l: int, aux: int):
            nonlocal filled_ctx_n, filled_aux_n
            if l < 0 or l >= len(file.file_lines):
                return False
            if file.file_lines_toks[l] is not None:
                return False
            t = self.enc.encode(file.file_lines[l])
            take_line = False
            if aux:
                if filled_aux_n + len(t) < limit_aux_n:
                    filled_aux_n += len(t)
                    er.check_aux_n += len(t)
                    take_line = True
            else:
                if filled_ctx_n + len(t) < limit_ctx_n + (limit_aux_n - filled_aux_n):
                    filled_ctx_n += len(t)
                    er.check_ctx_n += len(t)
                    take_line = True
            if not take_line:
                return False
            file.check_lines += 1
            file.file_lines_toks[l] = t
            if file.check_lines % LINE_NUMBER_EACH == 0:
                if aux:
                    filled_aux_n += toks_count_LINE
                else:
                    filled_ctx_n += toks_count_LINE
            return True

        def expand_FILE(i, file: _File) -> bool:
            nonlocal filled_ctx_n, filled_aux_n
            anything_works = False
            for ri, er in enumerate(file.expanding_ranges):
                if er.works0:
                    if er.line0 - 1 > 0 and file.file_lines_toks[er.line0 - 1] is not None:
                        # We bumped into another expanding range
                        print(" ! bumped into another expanding range er.line0 - 1 = %d" % (er.line0 - 1))
                        er.need_header = False
                        if er.aux:
                            filled_aux_n -= toks_count_LINE
                        else:
                            filled_ctx_n -= toks_count_LINE
                        er.works0 = False
                    success = _file_line2toks_helper(file, er, er.line0 - 1, aux=er.aux)
                    if success:
                        er.line0 -= 1
                    else:
                        er.works0 = False
                if er.works1:
                    success = _file_line2toks_helper(file, er, er.line1 + 1, aux=er.aux)  # For example we start with the range (5, 5) and expand from there, the line below is 6
                    if success and er.line1 + 1 >= len(file.file_lines) - 1:
                        er.works1 = False
                        er.line1 = len(file.file_lines) - 1
                    elif success:
                        er.line1 += 1
                        assert er.line1 < len(file.file_lines), ri
                    else:
                        er.works1 = False
                print("range%d: %d..%d, %d, %d, aux=%d, need_header=%i" % (ri, er.line0, er.line1, er.works0, er.works1, er.aux, er.need_header))
                anything_works |= er.works0 or er.works1
                recheck_ctx_n = 0
                recheck_aux_n = 0
                for line in range(er.line0, er.line1 + 1):
                    if er.aux:
                        recheck_aux_n += len(file.file_lines_toks[line])
                    else:
                        recheck_ctx_n += len(file.file_lines_toks[line])
                assert recheck_aux_n == er.check_aux_n
                assert recheck_ctx_n == er.check_ctx_n
            return anything_works

        def finish_FILE(i, file: _File):
            t, m = [], []
            assert len(file.file_lines) == len(file.file_lines_toks)
            line_countdown = 0
            first_header = True
            for line_n, line_toks in enumerate(file.file_lines_toks):
                if not line_toks:
                    line_countdown = 0
                    continue
                if line_countdown == 0:
                    line_n_t = [self.enc.ESCAPE] + self.enc.encode("LINE%04d\n" % (line_n + file.formal_line0))
                    print("------ len of line_n_t = %d == %d" % (len(line_n_t), toks_count_LINE))
                    t.extend(line_n_t)
                    m.extend([1 if not first_header else 0]*len(line_n_t))
                    first_header = False
                    line_countdown = 15
                t.extend(line_toks)
                m.extend([1]*len(line_toks))
                line_countdown -= 1
            plan_toks[i].extend(t)
            plan_mask[i].extend(m)
            plan_toks[i].extend(file.footer_toks)
            plan_mask[i].extend([1]*len(file.footer_toks))

        filled_ctx_n = 2   # ESCAPE, EOT in the end
        filled_aux_n = 0
        switch_init = {"MSG": dump_MSG, "FILE": init_FILE, "CHUNK": dump_CHUNK}
        switch_expand = {"FILE": expand_FILE}
        switch_finish = {"FILE": finish_FILE}
        for i, p in enumerate(self.plan):
            filled_ctx_n += 1   # ESCAPE
            switch_init[p.el_type](i, p)   # type: ignore
            print("%i %s init plan %i tokens" % (i, p.el_type, len(plan_toks[i])))
            print(termcolor.colored(self.enc.decode(plan_toks[i]), "red"))
            filled_ctx_n += len(plan_toks[i])
        print("after init, filled_ctx_n %d < limit_ctx_n %d, limit_aux_n %d, limit_ctx_n+limit_aux_n %d" % (filled_ctx_n, limit_ctx_n, limit_aux_n, limit_ctx_n + limit_aux_n))
        if filled_ctx_n > limit_ctx_n:
            excess = filled_ctx_n - limit_ctx_n
            limit_aux_n = max(0, limit_aux_n - excess)
            print("WARNING: initial filled_ctx_n %d > limit_ctx_n %d. Reduced limit_aux_n to %d" % (filled_ctx_n, limit_ctx_n, limit_aux_n))
        while 1:
            any_still_expanding = False
            for i, p in enumerate(self.plan):
                if p.el_type not in switch_expand:
                    continue
                print("expand %i %s" % (i, p.el_type), "filled_ctx_n %d < %d" % (filled_ctx_n, limit_ctx_n),  "filled_aux_n %d < %d" % (filled_aux_n, limit_aux_n))
                any_still_expanding |= switch_expand[p.el_type](i, p)  # type: ignore
                print(
                    #" => ctx %i aux %i lines %i" % (p.check_ctx_n, p.check_aux_n, p.check_lines),
                    " total ctx %i aux %i," % (filled_ctx_n, filled_aux_n),
                    " projected ctx_n+aux_n %i," % (filled_ctx_n + filled_aux_n),
                )
            if not any_still_expanding:
                break
        for i, p in enumerate(self.plan):
            if p.el_type not in switch_finish:
                continue
            switch_finish[p.el_type](i, p)  # type: ignore
        for i, (toks, msk) in enumerate(zip(plan_toks, plan_mask)):
            self.r.append(self.enc.ESCAPE)
            self.m.append(1 if i >= mask_from_plan_n else 0)
            self.r.extend(toks)
            self.m.extend(msk if i >= mask_from_plan_n else [0]*len(msk))
        self.r.extend([self.enc.ESCAPE, self.enc.EOT])
        self.m.extend([1, 1])
        assert len(self.r) == len(self.m)
        print("projected filled_ctx_n %d < limit %d" % (filled_ctx_n, limit_ctx_n))
        print("projected filled_aux_n %d < limit %d" % (filled_aux_n, limit_aux_n))
        print("projected filled_ctx_n+filled_aux_n = %d < %d" % (filled_ctx_n + filled_aux_n, limit_ctx_n + limit_aux_n))
        print("                       real context = %d" % (len(self.r),))
        return filled_ctx_n, filled_aux_n

    def dump_r(self):
        return hlprint(enc, self.r, self.m)

    def __repr__(self) -> str:
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


def test_messages(enc: SMCEncoding):
    t = Contrast2023q2(enc)
    t.add_msg("SYSTEM", "You are a coding assistant.")
    t.add_msg("USER", "how are you?")
    t.add_msg("ASSISTANT", "I'm not sure, I think I have bugs.")
    limit_ctx_n = 100
    limit_aux_n = 0
    filled_ctx_n, filled_aux_n = t.pack_context(0, limit_ctx_n, limit_aux_n)
    print(hlprint(enc, t.r, t.m))
    assert filled_ctx_n == len(t.r)
    assert filled_aux_n == 0


def test_expansion(enc: SMCEncoding):
    orig = ["# this is line %d" % i for i in range(30)]
    dest = orig[:]
    dest[10] = "# changed line"
    external_poi_ranges: Optional[DefaultDict[str, List[Tuple[int, int]]]] = None
    external_poi_ranges = defaultdict(list)
    external_poi_ranges["test.py"] = [(20, 20), (25, 25)]
    odm = {
        "orig": {
            'test.py': "\n".join(orig),
        },
        "dest": {
            'test.py': "\n".join(dest),
        },
        "commitmsg": "Expansion test",
    }
    for n_ctx in range(200, 400, 100):
        t = Contrast2023q2(enc)
        limit_aux_n = 100
        limit_ctx_n = n_ctx - limit_aux_n
        t.from_odm_dict(odm, limit_ctx_n, limit_aux_n, tight_shrink=True, external_poi_ranges=external_poi_ranges)
        print(t.dump_r())
        time.sleep(1)
        quit()



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
    # test_messages(enc)
    test_expansion(enc)
    # self_test(enc, example_odm, verbose=True, limit_ctx_n=512, limit_aux_n=128)

