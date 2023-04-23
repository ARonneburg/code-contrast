import random
import time
import termcolor

from cdifflib import CSequenceMatcher

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.contrast.contrast_stochastic import ops_remove_short_equals
from code_contrast.contrast.contrast_stochastic import ops_stochastic_expand
from code_contrast.print_utils import editclass_print, hlprint

from collections import defaultdict
from dataclasses import dataclass, field

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


element_types = ["SYSTEM", "USER", "ASSISTANT", "FILE", "CHUNK", "TOOL", "OUTPUT"]


ADDITIONAL_CHECKS = True


@dataclass
class _PlanElement:
    el_type: str


@dataclass
class _FileExpandingRange:  # Only for prompt packing
    aux: int
    line0: int
    line1: int
    line0expand: int = -1
    line1expand: int = -1
    works0: bool = True
    works1: bool = True


@dataclass
class _File(_PlanElement):
    file_fn: str
    file_lines: List[str]
    formal_line0: int = -1  # §LINE1337 first line
    # For prompt packing:
    file_lines_toks: List[Optional[List[int]]] = field(default_factory=lambda: [])
    expanding_ranges: List[_FileExpandingRange] = field(default_factory=lambda: [])
    footer_toks: List[int] = field(default_factory=lambda: [])
    lineheaders_dirty: bool = True
    lineheaders_cnt_n: int = 0
    lineheaders_aux_n: int = 0


@dataclass
class _Chunk(_PlanElement):
    orig_file: _File
    # For prompt generation:
    dest_text: List[str]
    i0: int
    i1: int
    j0: int
    j1: int
    formal_line: int = -1
    shift: int = -1
    # For decode:
    to_del: List[str] = field(default_factory=lambda: [])
    to_ins: List[str] = field(default_factory=lambda: [])
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
        assert msg_role in element_types
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
            filled_aux_n += len(file.footer_toks)
            # Each range has a header, until it bumps into another range above when exanding
            file.file_lines_toks = [None] * len(file.file_lines)
            file.lineheaders_cnt_n = 0
            file.lineheaders_aux_n = 0
            file.lineheaders_dirty = True
            for er in file.expanding_ranges:
                er.works0 = True
                er.works1 = True
                er.line0expand = er.line0
                er.line1expand = er.line1
                for line in range(er.line0expand, er.line1expand + 1):
                    _file_line2toks_helper(file, er, line, aux=er.aux)
            _file_lineheader_tokens(file)

        def _file_lineheader_tokens(file: _File):
            nonlocal filled_ctx_n, filled_aux_n
            if not file.lineheaders_dirty:
                return
            # Intersecting ranges will make the estimation larger than it should be, causing this
            # calculation to be more conservative => the end result is a less filled context.
            cnt_lineheaders_n = sum(
                1 + (er.line1expand - er.line0expand + 1) // LINE_NUMBER_EACH
                for er in file.expanding_ranges if not er.aux
            )
            aux_lineheaders_n = sum(
                1 + (er.line1expand - er.line0expand + 1) // LINE_NUMBER_EACH
                for er in file.expanding_ranges if er.aux
            )
            file.lineheaders_dirty = False
            if cnt_lineheaders_n != file.lineheaders_cnt_n:
                filled_ctx_n += (cnt_lineheaders_n - file.lineheaders_cnt_n) * toks_count_LINE
                file.lineheaders_cnt_n = cnt_lineheaders_n
            if aux_lineheaders_n != file.lineheaders_aux_n:
                filled_aux_n += (aux_lineheaders_n - file.lineheaders_aux_n) * toks_count_LINE
                file.lineheaders_aux_n = aux_lineheaders_n

        def _file_line2toks_helper(file: _File, er: _FileExpandingRange, l: int, aux: int):
            nonlocal filled_ctx_n, filled_aux_n
            _file_lineheader_tokens(file)
            if l < 0 or l >= len(file.file_lines):
                return False
            if file.file_lines_toks[l] is not None:
                return False
            t = self.enc.encode(file.file_lines[l])
            take_line = False
            if aux:
                if filled_aux_n + len(t) < limit_aux_n:
                    # print("take aux line %i" % (l))
                    filled_aux_n += len(t)
                    take_line = True
            else:
                if filled_ctx_n + len(t) < limit_ctx_n + (limit_aux_n - filled_aux_n):
                    # print("take ctx line %i" % (l))
                    filled_ctx_n += len(t)
                    take_line = True
            if not take_line:
                return False
            file.file_lines_toks[l] = t
            file.lineheaders_dirty = True
            return True

        def expand_FILE(i, file: _File, aux) -> bool:
            anything_works = False
            for ri, er in enumerate(file.expanding_ranges):
                if er.aux != aux:
                    continue
                if er.works0:
                    # if er.line0expand - 1 > 0 and file.file_lines_toks[er.line0expand - 1] is not None:
                    #     print(" ! bumped into another expanding range er.line0expand - 1 = %d" % (er.line0expand - 1))
                    #     er.works0 = False
                    success = _file_line2toks_helper(file, er, er.line0expand - 1, aux=er.aux)
                    if success:
                        er.line0expand -= 1
                    else:
                        er.works0 = False
                if er.works1:
                    success = _file_line2toks_helper(file, er, er.line1expand + 1, aux=er.aux)  # For example we start with the range (5, 5) and expand from there, the line below is 6
                    if success and er.line1expand + 1 >= len(file.file_lines) - 1:
                        er.works1 = False
                        er.line1expand = len(file.file_lines) - 1
                    elif success:
                        er.line1expand += 1
                        assert er.line1expand < len(file.file_lines), ri
                    else:
                        er.works1 = False
                # print("range%d: %d..%d, %d, %d, aux=%d, need_header=%i" % (ri, er.line0expand, er.line1expand, er.works0, er.works1, er.aux, er.need_header))
                anything_works |= er.works0 or er.works1
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
            filled_ctx_n += len(plan_toks[i])
        if filled_ctx_n > limit_ctx_n:
            excess = filled_ctx_n - limit_ctx_n
            limit_aux_n = max(0, limit_aux_n - excess)
            print("WARNING: initial filled_ctx_n %d > limit_ctx_n %d. Reduced limit_aux_n to %d" % (filled_ctx_n, limit_ctx_n, limit_aux_n))
        for aux in [1, 0]:
            while 1:
                any_still_expanding = False
                for i, p in enumerate(self.plan):
                    if p.el_type not in switch_expand:
                        continue
                    # print("expand %i %s" % (i, p.el_type), "filled_ctx_n %d < %d" % (filled_ctx_n, limit_ctx_n),  "filled_aux_n %d < %d" % (filled_aux_n, limit_aux_n))
                    any_still_expanding |= switch_expand[p.el_type](i, p, aux)  # type: ignore
                    # print(
                    #     " => total ctx %i aux %i," % (filled_ctx_n, filled_aux_n),
                    #     "projected ctx_n+aux_n %i\n" % (filled_ctx_n + filled_aux_n),
                    # )
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
        # print("projected filled_ctx_n %d < limit %d" % (filled_ctx_n, limit_ctx_n))
        # print("projected filled_aux_n %d < limit %d" % (filled_aux_n, limit_aux_n))
        # print("projected filled_ctx_n+filled_aux_n = %d < %d" % (filled_ctx_n + filled_aux_n, limit_ctx_n + limit_aux_n))
        # print("                       real context = %d" % (len(self.r),))
        assert len(self.r) == len(self.m)
        assert len(self.r) <= filled_ctx_n + filled_aux_n, "Packed tokens %d, upper bound on number of tokens %d. May be an internal bug, maybe toks_count_LINE is not the max value possible." % (len(self.r), filled_ctx_n + filled_aux_n)
        return filled_ctx_n, filled_aux_n

    def dump_r(self):
        return hlprint(self.enc, self.r, self.m)

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

