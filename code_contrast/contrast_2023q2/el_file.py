import random
import time
import termcolor

from cdifflib import CSequenceMatcher

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.contrast.contrast_stochastic import ops_remove_short_equals
from code_contrast.contrast.contrast_stochastic import ops_stochastic_expand
from code_contrast.print_utils import editclass_print, hlprint
from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, element_classes

from collections import defaultdict
from dataclasses import dataclass, field

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


@dataclass
class _FileExpandingRange:
    line0: int
    line1: int
    aux: int
    line0expand: int = -1
    line1expand: int = -1
    works0: bool = True
    works1: bool = True


class FileElement(Element):
    def __init__(self, file_fn: str, file_lines: List[str], LINE_NUMBER_EACH=15):
        super().__init__("FILE")
        self.file_fn = file_fn
        self.file_lines = file_lines
        self.file_lines_toks: List[Optional[List[int]]] = []
        self.footer_toks = list()
        self.lineheaders_dirty = True
        self.lineheaders_cnt_n = 0
        self.lineheaders_aux_n = 0
        self.toks_count_LINE = -1
        self.expanding_ranges: List[_FileExpandingRange] = list()
        self.LINE_NUMBER_EACH = LINE_NUMBER_EACH

    def add_expanding_range(self, line0: int, line1: int, aux: int):
        self.expanding_ranges.append(_FileExpandingRange(
            line0=max(0, min(line0, len(self.file_lines) - 1)),
            line1=max(0, min(line1, len(self.file_lines) - 1)),
            aux=aux))

    def pack_init(self, cx: ElementPackingContext) -> Tuple[List[int], List[int]]:
        header_toks = cx.enc.encode("FILE " + self.file_fn.replace("\n", "\\n") + "\n")
        self.toks_count_LINE = len([cx.enc.ESCAPE] + cx.enc.encode("LINE%04d\n" % 1234))
        self.footer_toks = [cx.enc.ESCAPE] + cx.enc.encode("/FILE\n")
        cx.filled_aux_n += len(self.footer_toks)
        t, m = [], []
        t.extend(header_toks)
        m.extend([1]*len(header_toks))
        # Each range has a header, until it bumps into another range above when exanding
        self.file_lines_toks = [None] * len(self.file_lines)
        self.lineheaders_dirty = True
        self.lineheaders_cnt_n = 0
        self.lineheaders_aux_n = 0
        for er in self.expanding_ranges:
            er.works0 = True
            er.works1 = True
            er.line0expand = er.line0
            er.line1expand = er.line1
            for line in range(er.line0expand, er.line1expand + 1):
                self._lines2toks_helper(cx, line, aux=er.aux)
        self._estimate_line_header_tokens(cx)
        return t, m

    def _estimate_line_header_tokens(self, cx: ElementPackingContext):
        if not self.lineheaders_dirty:
            return
        # Intersecting ranges will make the estimation larger than it should be, causing this
        # calculation to be more conservative => the end result is a less filled context.
        cnt_lineheaders_n = sum(
            1 + (er.line1expand - er.line0expand + 1) // self.LINE_NUMBER_EACH
            for er in self.expanding_ranges if not er.aux
        )
        aux_lineheaders_n = sum(
            1 + (er.line1expand - er.line0expand + 1) // self.LINE_NUMBER_EACH
            for er in self.expanding_ranges if er.aux
        )
        self.lineheaders_dirty = False
        if cnt_lineheaders_n != self.lineheaders_cnt_n:
            cx.filled_ctx_n += (cnt_lineheaders_n - self.lineheaders_cnt_n) * self.toks_count_LINE
            self.lineheaders_cnt_n = cnt_lineheaders_n
        if aux_lineheaders_n != self.lineheaders_aux_n:
            cx.filled_aux_n += (aux_lineheaders_n - self.lineheaders_aux_n) * self.toks_count_LINE
            self.lineheaders_aux_n = aux_lineheaders_n

    def _lines2toks_helper(self, cx: ElementPackingContext, l: int, aux: int):
        self._estimate_line_header_tokens(cx)
        if l < 0 or l >= len(self.file_lines):
            return False
        if self.file_lines_toks[l] is not None:
            return False
        t = cx.enc.encode(self.file_lines[l])
        take_line = False
        if aux:
            if cx.filled_aux_n + len(t) < cx.limit_aux_n:
                # print("take aux line %i" % (l))
                cx.filled_aux_n += len(t)
                take_line = True
        else:
            if cx.filled_ctx_n + len(t) < cx.limit_ctx_n + (cx.limit_aux_n - cx.filled_aux_n):
                # print("take ctx line %i" % (l))
                cx.filled_ctx_n += len(t)
                take_line = True
        if not take_line:
            return False
        self.file_lines_toks[l] = t
        self.lineheaders_dirty = True
        return True

    def pack_inflate(self, cx: ElementPackingContext, aux: bool) -> bool:
        anything_works = False
        for ri, er in enumerate(self.expanding_ranges):
            if er.aux != aux:
                continue
            if er.works0:
                # if er.line0expand - 1 > 0 and self.file_lines_toks[er.line0expand - 1] is not None:
                #     print(" ! bumped into another expanding range er.line0expand - 1 = %d" % (er.line0expand - 1))
                #     er.works0 = False
                success = self._lines2toks_helper(cx, er.line0expand - 1, aux=er.aux)
                if success:
                    er.line0expand -= 1
                else:
                    er.works0 = False
            if er.works1:
                success = self._lines2toks_helper(cx, er.line1expand + 1, aux=er.aux)  # For example we start with the range (5, 5) and expand from there, the line below is 6
                if success and er.line1expand + 1 >= len(self.file_lines) - 1:
                    er.works1 = False
                    er.line1expand = len(self.file_lines) - 1
                elif success:
                    er.line1expand += 1
                    assert er.line1expand < len(self.file_lines), ri
                else:
                    er.works1 = False
            # print("range%d: %d..%d, %d, %d, aux=%d, need_header=%i" % (ri, er.line0expand, er.line1expand, er.works0, er.works1, er.aux, er.need_header))
            anything_works |= er.works0 or er.works1
        return anything_works

    def pack_finish(self, cx: ElementPackingContext) -> Tuple[List[int], List[int]]:
        t, m = [], []
        assert len(self.file_lines) == len(self.file_lines_toks)
        line_countdown = 0
        first_header = True
        for line_n, line_toks in enumerate(self.file_lines_toks):
            if not line_toks:
                line_countdown = 0
                continue
            if line_countdown == 0:
                line_n_t = [cx.enc.ESCAPE] + cx.enc.encode("LINE%04d\n" % (line_n + self.formal_line0))
                t.extend(line_n_t)
                m.extend([1 if not first_header else 0]*len(line_n_t))
                first_header = False
                line_countdown = 15
            t.extend(line_toks)
            m.extend([1]*len(line_toks))
            line_countdown -= 1
        t.extend(self.footer_toks)
        m.extend([1]*len(self.footer_toks))
        return t, m
