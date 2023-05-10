import random
import time
from code_contrast.contrast_2023q2.element import Format2023q2
import termcolor

from cdifflib import CSequenceMatcher

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.contrast.contrast_stochastic import ops_remove_short_equals
from code_contrast.contrast.contrast_stochastic import ops_stochastic_expand
from code_contrast.print_utils import editclass_print, hlprint

from collections import defaultdict
from dataclasses import dataclass, field

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional

from code_contrast.contrast_2023q2.packing import Packer
from code_contrast.contrast_2023q2.el_file import FileElement
from code_contrast.contrast_2023q2.el_chunk import ChunkElement
from code_contrast.contrast_2023q2.el_msg import MsgElement


def from_odm_dict(
    fmt: Format2023q2,
    odm: Dict[str, Any],
    # random_shrink = True,
    tight_shrink = False,
    exact_cx_lines0 = -1,
    exact_cx_lines1 = -1,
    external_poi_ranges: Optional[DefaultDict[str, List[Tuple[int, int]]]] = None,
) -> Packer:
    pack = Packer(fmt)
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
    for fn in fns:
        f = FileElement(fn, [(x + "\n") for x in odm["orig"][fn].splitlines()])
        pack.add_to_plan(f)
        if external_poi_ranges and fn in external_poi_ranges:
            poi_list = external_poi_ranges[fn]
            for line0, line1 in poi_list:
                f.add_expanding_range(line0, line1, aux=1)
        files.append(f)
    msg = MsgElement("USER", odm["commitmsg"])
    pack.add_to_plan(msg)
    for fn, f in zip(fns, files):
        chunks.extend(_run_diff_for_single_file(f, [(x + "\n") for x in odm["dest"][fn].splitlines()], exact_cx_lines0, exact_cx_lines1))
    random.shuffle(chunks)
    for chunk in chunks:
        pack.add_to_plan(chunk)
    return pack


def _run_diff_for_single_file(f: FileElement, dest_text: List[str], exact_cx_lines0: int, exact_cx_lines1: int):
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
        c = ChunkElement(f)
        c.assign_from_diff(dest_text[j0:j1], i0, i1, j0, j1)
        chunks.append(c)
        f.add_expanding_range(line0=i0, line1=i1-1, aux=0)
    return chunks
