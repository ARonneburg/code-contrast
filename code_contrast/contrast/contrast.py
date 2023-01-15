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
from code_contrast.print_utils import editclass_print

from collections import defaultdict
from dataclasses import dataclass

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


OFFSET_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TPOS_HIGH_WATER = len(OFFSET_CHARS)   # TODO: not necessary anymore, remove and run test
TPOS_LOW_WATER = 16


@dataclass
class Edit:
    fn: str
    tpos: int
    shift: int
    todel: List[int]
    toins: List[int]
    i1: int = -1
    i2: int = -1
    real_delstart: int = -1
    real_delends: int = -1
    error: str = ""
    fuzzy: int = -1


class DecodeError(ValueError):
    pass


class TooBig(ValueError):
    pass


WAIT, FILENAME, FILENAME_DIAMONDS, CODE, CODE_FINISHING, MSG, CHUNK, DEL, SHIFT, INS = range(10)


def parse_fn(fn: str) -> Tuple[str, Optional[int]]:
    if re.search(r":([0-9]+)$", fn) is None:
        return fn, None
    return ":".join(fn.split(":")[:-1]), int(fn.split(":")[-1])


class UntokenizeState:
    def __init__(self):    # full_orig_tokens: Dict[str, List[int]], orig_withpos: Dict[str, List[int]]):
        self.state = WAIT
        self.c = 0
        self.brewing_edit: Edit = Edit("", 0, -1, [], [])
        self.fn_tokens = list()
        self.fn_txt = ""
        self.body_tokens = list()
        self.msg_tokens = list()
        self.eot = False
        self.stats = {
            "chunks_applied": 0,
            "files_unchanged": 0,
            "files_patched": 0,
            "invalid_tpos": 0,
            "errors": 0,
            "fuzzy": 0,
            # "tokens": len(self.r),
            }
        self.scratch: Dict[str, List[int]] = dict()
        self.orig2scratch: Dict[str, List[int]] = dict()


class ContrastDiff:
    def __init__(self, enc: SMCEncoding):
        self.enc: SMCEncoding = enc
        self.orig_tokens: Dict[str, List[int]] = dict()
        self.orig_withpos: Dict[str, List[int]] = dict()
        self.dest_tokens: Dict[str, List[int]] = dict()
        self.commitmsg: str = ""
        self.edits: List[Edit] = list()
        self.fn2tind: DefaultDict[str, List[int]] = defaultdict(list)
        self.fn2tstart: Dict[str, int] = dict()
        self.fn2cut0: Dict[str, int] = dict()
        self.r: List[int] = list()
        self.m: List[int] = list()
        self.errors: List[str] = list()
        self.tokens_without_shortening = -1
        self.tokens_without_edits = -1
        self.offset_commitmsg = -1
        self.offset_code_start = -1
        self.offset_code_end = -1
        self.offset_edits = -1
        self.file_deltokens: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.file_dellines: Dict[str, List[int]] = defaultdict(list)
        self.file_contlines: Dict[str, List[int]] = defaultdict(list)

    def from_odm_dict(
        self,
        odm: Dict[str, Any],
        n_ctx: int,
        commit_ahead: bool =True,
        contrast_unmask_orig: int = 0,
        auto_shrink = True,
        tight_shrink = False,
        exact_cx_lines0 = -1,
        exact_cx_lines1 = -1,
    ) -> Dict[str, List[int]]:
        files1 = set(odm["orig"].keys()) if ("orig" in odm) else set(odm["orig_tokens"].keys())
        files2 = set(odm["dest"].keys()) if ("dest" in odm) else set(odm["dest_tokens"].keys())
        assert files1 == files2
        files = list(files1)
        random.shuffle(files)
        file_poi: Dict[str, Set[int]] = defaultdict(set)
        file_deltokens = defaultdict(list)
        file_dellines = defaultdict(list)
        file_contlines = defaultdict(list)
        opblocks = []
        for fn in files:
            assert ("orig_tokens" in odm and "dest_tokens" in odm) or ("orig" in odm and "dest" in odm)
            # Doesn't work well with \u2028
            # orig_lines = odm["orig"][fn].replace('\r\n', '\n').replace('\r', '\n').splitlines()
            # dest_lines = odm["dest"][fn].replace('\r\n', '\n').replace('\r', '\n').splitlines()
            orig_lines = [x+"\n" for x in odm["orig"][fn].splitlines()]
            dest_lines = [x+"\n" for x in odm["dest"][fn].splitlines()]
            if len(orig_lines)==0:
                orig_lines.append("\n")
            if orig_lines[-1][-1] != "\n":
                orig_lines[-1] += "\n"
            if dest_lines[-1][-1] != "\n":
                dest_lines[-1] += "\n"
            orig_all_tokens = []
            dest_all_tokens = []
            fndiff = list(CSequenceMatcher(None, orig_lines, dest_lines).get_opcodes())
            dellines = []
            contlines = []
            for tag, i1, i2, j1, j2 in fndiff:
                if tag in ["replace", "delete", "insert"]:
                    dellines.append(i1)
                    contlines.extend(list(range(i1+1, i2)))
            fndiff = ops_stochastic_expand(fndiff,
                left_prob=1, right_prob=1,
                exact_cx_lines0=exact_cx_lines0, exact_cx_lines1=exact_cx_lines1,
                disable_insert=True)
            fndiff = ops_remove_short_equals(fndiff, upto=2)
            DUMP_DIFF = 0
            def orig_app(line):
                tmp = self.enc.encode(line)
                if i in dellines:
                    file_dellines[fn].append(len(orig_all_tokens) + len(tmp) - 1)
                    file_poi[fn].add(len(orig_all_tokens) + len(tmp))
                if i in contlines:
                    file_contlines[fn].append(len(orig_all_tokens) + len(tmp) - 1)
                    file_poi[fn].add(len(orig_all_tokens) + len(tmp))
                orig_all_tokens.extend(tmp)
                block_orig_t.extend(tmp)
                return tmp
            for tag, i1, i2, j1, j2 in fndiff:
                block_orig_t = []
                block_dest_t = []
                if tag == "equal":
                    # equal
                    assert orig_lines[i1:i2] == dest_lines[j1:j2]
                    for i in range(i1, i2):
                        if DUMP_DIFF:
                            print("%05i" % len(orig_all_tokens), "%04i" % i, " ", orig_lines[i].rstrip("\n"))
                        tmp = orig_app(orig_lines[i])
                        dest_all_tokens.extend(tmp)
                        assert tmp[-1] == 198, tmp
                    continue
                # not equal
                assert tag in ["replace", "delete", "insert", "joined"], tag
                i_shift = len(orig_all_tokens)
                j_shift = len(dest_all_tokens)
                for i in range(i1, i2):
                    if DUMP_DIFF:
                        print("%05i" % len(orig_all_tokens), "%04i" % i, "-", orig_lines[i].rstrip("\n"))
                    orig_app(orig_lines[i])
                if DUMP_DIFF:
                    for j in range(j1, j2):
                        print("     ", "%04i" % i, "+", dest_lines[j].rstrip("\n"))
                for line in dest_lines[j1:j2]:
                    tmp = self.enc.encode(line)
                    dest_all_tokens.extend(tmp)
                    block_dest_t.extend(tmp)
                for test in [self.enc.DIAMOND, self.enc.CHUNK, self.enc.ESCAPE]:
                    assert test not in block_orig_t, "token %i found in orig tokens" % test
                # highlight only
                patch = list(CSequenceMatcher(None, block_orig_t, block_dest_t, autojunk=False).get_opcodes())
                patch = ops_remove_short_equals(patch, upto=3)
                for op, ti1, ti2, tj1, tj2 in patch:
                    if op == "equal":
                        continue
                    file_deltokens[fn].append((i_shift + ti1, i_shift + ti2))
                # changed the whole block
                patch = [("replace", 0, len(block_orig_t), 0, len(block_dest_t))]
                opblock = []
                for op, ti1, ti2, tj1, tj2 in patch:
                    if op == "equal":
                        continue
                    opblock.append((fn, op, i_shift + ti1, i_shift + ti2, j_shift + tj1, j_shift + tj2))
                    file_poi[fn].add(i_shift + ti1)
                    file_poi[fn].add(i_shift + ti2)
                opblocks.append(opblock)
            self.orig_tokens[fn] = orig_all_tokens
            self.dest_tokens[fn] = dest_all_tokens
        random.shuffle(opblocks)
        raw_ops: List[Tuple[str, str, int, int, int, int]] = list()
        for opblock in opblocks:
            raw_ops.extend(opblock)
        commitmsg_tokens = [self.enc.MSG] + self.enc.encode(" " + odm["commitmsg"])

        def generate_edits():
            self.edits = []
            for fn, _op, i1, i2, j1, j2 in raw_ops:
                cut = self.fn2cut[fn]
                orig_t = self.orig_tokens[fn][cut:]
                dest_t = self.dest_tokens[fn]
                i1 -= cut
                i2 -= cut
                starts = self.fn2tstart[fn]
                tinds = self.fn2tind[fn]
                written_i1 = tinds.index(i1)   # index in r is 'written_i1' plus 'starts'
                written_i2 = tinds.index(i2)
                written_tpos = 0
                for deli1, deli2 in file_deltokens[fn]:
                    written_deli1 = tinds.index(deli1 - cut)
                    if written_i1 <= written_deli1 <= written_i2:  # Use the first position token after the real changes start
                        written_tpos = written_deli1
                        break
                while tinds[written_tpos] != -1:
                    written_tpos += 1
                    assert written_tpos < len(tinds)
                assert self.enc.is_tpos(self.r[starts + written_tpos]) or self.r[starts + written_tpos]==0
                ahead_newlines = 0
                for i in range(written_i1, written_tpos):
                    if self.r[starts + i] == self.enc.LF:
                        ahead_newlines += 1
                # skip_tokens = written_tpos - written_i1
                assert 0 <= ahead_newlines < TPOS_HIGH_WATER
                self.edits.append(Edit(
                    fn,
                    self.r[starts + written_tpos],
                    ahead_newlines,
                    [orig_t[i] for i in range(i1, i2)],
                    [dest_t[j] for j in range(j1, j2)],
                    i1=written_i1 + starts,
                    i2=written_i2 + starts,
                ))
            self.file_deltokens = defaultdict(list)
            for fn in file_deltokens.keys():
                cut = self.fn2cut[fn]
                starts = self.fn2tstart[fn]
                tinds = self.fn2tind[fn]
                for i1, i2 in file_deltokens[fn]:
                    written_i1 = tinds.index(i1 - cut)
                    written_i2 = tinds.index(i2 - cut)
                    self.file_deltokens[fn].append((written_i1 + starts, written_i2 + starts))
            self.file_dellines = defaultdict(list)
            self.file_contlines = defaultdict(list)
            for fn in file_dellines.keys():
                cut = self.fn2cut[fn]
                starts = self.fn2tstart[fn]
                tinds = self.fn2tind[fn]
                # orig_t = self.orig_tokens[fn][cut:]
                for ti in file_dellines[fn]:
                    if (ti - cut) not in tinds:
                        print("file_dellines[%s]" % fn, file_dellines[fn])
                        print("orig_tokens[fn]", len(self.enc.decode(self.orig_tokens[fn])))
                        print("tinds", len(tinds))
                        print("poi", file_poi[fn])
                        assert 0
                    written_i = tinds.index(ti - cut)
                    self.file_dellines[fn].append(written_i + starts)
                for ti in file_contlines[fn]:
                    written_i = tinds.index(ti - cut)
                    self.file_contlines[fn].append(written_i + starts)

        def append_with_tpos_tokens(tlist: List[int], fn: str):
            # while len(self.r) % 16 != 0:
            #     self.r.append(self.enc.DIAMOND)
            #     self.m.append(0)
            self.fn2tstart[fn] = len(self.r)
            without_tpos = 0
            cursor = 0
            tinds = self.fn2tind[fn]
            def app(t, m):
                self.r.append(t)
                self.m.append(m)
                tinds.append(cursor if m else -1)
            for t in tlist:
                # if without_tpos == TPOS_HIGH_WATER-1 or (without_tpos > TPOS_LOW_WATER and t==self.enc.LF):
                if t==self.enc.LF and without_tpos > TPOS_LOW_WATER:
                    app(tpos_unused.pop() if len(tpos_unused) > 0 else 0, 0)
                    without_tpos = 0
                app(t, 1)
                without_tpos += 1
                cursor += 1
            app(tpos_unused.pop() if len(tpos_unused) > 0 else 0, 0)

        passes = ["est", "real"] if (auto_shrink or tight_shrink) else ["real"]
        for pas in passes:
            self.r = []
            self.m = []
            if commit_ahead:
                self.offset_commitmsg = len(self.r) + 2
                self.r.extend(commitmsg_tokens)
                self.m.extend([0]*len(commitmsg_tokens))
            self.fn2tind = defaultdict(list)
            self.fn2tstart = dict()
            self.fn2cut = dict()
            tpos_unused = list(self.enc.tpos)
            random.shuffle(tpos_unused)
            tpos_unused *= 2
            need_to_cut = 0
            if self.tokens_without_shortening > n_ctx:
                need_to_cut = self.tokens_without_shortening - n_ctx
            # print("tight_shrink=%i need_to_cut=%i, self.tokens_without_edits=%i, self.tokens_without_shortening=%i, n_ctx=%i" % (
            #    tight_shrink, need_to_cut, self.tokens_without_edits, self.tokens_without_shortening, n_ctx))
            #print("n_ctx=%i" % (n_ctx))
            saved_check = []
            self.offset_code_start = len(self.r)
            for fi, fn in enumerate(files):
                orig_t = self.orig_tokens[fn]
                i1, i2 = len(orig_t)//2, len(orig_t)//2
                if fn in file_poi:
                    i1, i2 = min(file_poi[fn]), max(file_poi[fn])
                r1, r2 = 0, len(orig_t)
                cut_this_file = need_to_cut - sum(saved_check)
                if pas == "real" and cut_this_file > 0:
                    for i in range(3):
                        if tight_shrink:
                            if i == 0:
                                #print("%s need_to_cut=%i, cut_this_file=%i, r1=%i, r2=%i (1)" % (fn, need_to_cut, cut_this_file, r1, r2))
                                r2 = max(i2, r2 - cut_this_file)
                            elif i == 1:
                                #print("%s need_to_cut=%i, cut_this_file=%i, r1=%i, r2=%i (2)" % (fn, need_to_cut, cut_this_file, r1, r2))
                                r1 = min(r1 + cut_this_file, i1)
                        else:
                            if random.random() < 0.5:
                                r1 = random.randint(r1, i1)
                            if random.random() < 0.5:
                                r2 = random.randint(i2, r2)
                        saved1 = r1
                        saved2 = len(orig_t) - r2
                        assert saved1 >= 0 and saved2 >= 0, f"i1={i1} i2={i2} r1={r1} r2={r2}"
                        cut_this_file = need_to_cut - saved1 - saved2 - sum(saved_check)
                        if cut_this_file <= 0:
                            break
                    # print(" => %s need_to_cut=%i, cut_this_file=%i, saved1=%i, saved2=%i" % (fn, need_to_cut, cut_this_file, saved1, saved2))
                    saved_check.append(saved1)
                    saved_check.append(saved2)
                t = [self.enc.FILE] + self.enc.encode(" " + fn + ":%i" % r1) + [self.enc.ESCAPE]
                self.r.extend(t)
                self.m.extend([0]*len(t))
                #print("writing %s %i..%i out of %i" % (fn, r1, r2, len(orig_t)))
                append_with_tpos_tokens(orig_t[r1:r2] + [self.enc.ESCAPE], fn)
                if pas=="real" and len(tpos_unused) < len(self.enc.tpos):
                    raise TooBig("too many position tokens was used")
                self.fn2cut[fn] = r1
            self.offset_code_end = len(self.r)
            if not commit_ahead:
                self.offset_commitmsg = len(self.r) + 2
                self.r.extend(commitmsg_tokens)
                self.m.extend([0]*len(commitmsg_tokens))
            #if self.tokens_without_shortening != -1:
            #    print("shouldbe tokens_without_shortening=%i - saved=%i = %i" % (self.tokens_without_shortening, sum(saved_check), self.tokens_without_shortening - sum(saved_check)))
            if pas == "est":
                generate_edits()
                self.tokens_without_edits = len(self.r)
                self.write_edits()
                self.tokens_without_shortening = len(self.r)
            else:
                generate_edits()
            # print("%s tpos_unused %i/%i" % (pas, len(tpos_unused) - len(self.enc.tpos), len(self.enc.tpos)))
        assert len(self.m) == len(self.r)
        self.code_m = self.m
        if not contrast_unmask_orig:
            self.m = [0]*len(self.m)
        return self.orig_tokens

    def dump_edits(self):
        acc = ""
        for e in self.edits:
            acc += " ".join(str(x) for x in [
                e.fn,
                e.tpos, self.enc.decode([e.tpos]), "-LF", e.shift,
                e.todel, "\"%s\"" % termcolor.colored(self.enc.decode(e.todel), "red").replace("\n", "\\n"),
                e.toins, "\"%s\"" % termcolor.colored(self.enc.decode(e.toins), "green").replace("\n", "\\n"),
                ])
            acc += "\n"
        return acc

    def write_edits(self):
        self.offset_edits = len(self.r)
        self.offset_first_postoken = -1
        self.r.extend([self.enc.CHUNK])
        self.m.extend([0]*1)
        # self.m.extend([int(self.offset_first_postoken != -1)]*2)
        for e in self.edits:
            if self.offset_first_postoken == -1:
                self.offset_first_postoken = len(self.r)
            self.r.append(e.tpos)
            self.r.extend(e.todel)
            self.r.append(self.enc.ESCAPE)
            self.m.extend([1] + [1]*len(e.todel) + [1])
            number_token = self.enc.encode(OFFSET_CHARS[e.shift])
            assert len(number_token) == 1
            self.r.append(number_token[0])
            self.r.extend(e.toins)
            self.r.append(self.enc.CHUNK)
            self.m.extend([1] + [1]*len(e.toins) + [1])
        self.r.append(self.enc.EOT)
        self.m.append(1)

    def write_esc_chunk(self):
        self.offset_edits = len(self.r)
        self.r.extend([self.enc.CHUNK])
        self.m.extend([0])
        self.offset_first_postoken = len(self.r)

    def edit_class_vector(self):
        class_vect = [0]*len(self.r)  # zero is no training
        for i in range(0, self.offset_code_end):
            if self.code_m[i]:
                # train "not edit"
                class_vect[i] = 1
                assert not self.enc.is_tpos(self.r[i])
        for e in self.edits:
            for i1, i2 in self.file_deltokens[e.fn]:
                for i in range(i1, i2):
                    if not self.enc.is_tpos(self.r[i]):
                        # train regular tokens "edit"
                        class_vect[i] = 2
        for fn in self.file_contlines.keys():
            for i in self.file_contlines[fn]:
                if self.r[i] == self.enc.LF:
                    # train end-of-line "continue"
                    class_vect[i] = 3
                else:
                    print("WARNING: not LF at %i/%i" % (i, len(class_vect)))
        for fn in self.file_dellines.keys():
            for i in self.file_dellines[fn]:
                if self.r[i] == self.enc.LF:
                    # train end-of-line "edit"
                    class_vect[i] = 2
                else:
                    print("WARNING: not LF at %i" % i)
        return class_vect

    def untokenize_init(self, full_orig_tokens: Dict[str, List[int]]):
        """
        Requires original, because files might be truncated to fit into context.
        """
        assert len(self.orig_tokens) == 0
        assert len(self.orig_withpos) == 0
        self.full_orig_tokens = full_orig_tokens
        us = UntokenizeState()   #orig_tokens, self.orig_withpos)
        return us

    def untokenize_finish_state(self, us: UntokenizeState, c: int):
        if us.state in [WAIT, FILENAME_DIAMONDS, DEL, SHIFT, CHUNK]:
            pass
        elif us.state == FILENAME:
            us.fn_txt = self.enc.decode(us.fn_tokens)
            # "file.py:25 "
            if len(us.fn_txt) > 0 and us.fn_txt[0] == " ":
                us.fn_txt = us.fn_txt[1:]
            us.fn_txt, shortened_tokens = parse_fn(us.fn_txt)
            if shortened_tokens is not None:
                if us.fn_txt in self.full_orig_tokens:
                    self.orig_tokens[us.fn_txt] = self.full_orig_tokens[us.fn_txt]
                    self.fn2cut0[us.fn_txt] = shortened_tokens
                    us.body_tokens = self.full_orig_tokens[us.fn_txt][:shortened_tokens]
                    # print("start body tokens %s: %s" % (fn_txt, body_tokens))
                else:
                    print(f"WARNING: '{us.fn_txt}' not found in the original")
            us.fn_tokens = list()
        elif us.state == CODE:
            assert 0
        elif us.state == CODE_FINISHING:
            # while len(body_tokens) > 0 and body_tokens[-1] == self.enc.ESCAPE:
            #     body_tokens.pop()
            if us.fn_txt:
                without_tpos = [t for t in us.body_tokens if not self.enc.is_tpos(t)]
                while len(without_tpos) and without_tpos[-1] in [self.enc.ESCAPE, self.enc.DIAMOND]:
                    without_tpos.pop(-1)
                if len(self.full_orig_tokens) > 0:  # else test mode
                    for i in range(len(without_tpos)):
                        assert without_tpos[i] == self.full_orig_tokens[us.fn_txt][i], "\n" + str(without_tpos) + "\n" + str(self.orig_tokens[us.fn_txt])
                if us.fn_txt in self.full_orig_tokens:
                    leftover_tokens = self.full_orig_tokens[us.fn_txt][len(without_tpos):]
                else:
                    leftover_tokens = []
                buf = us.body_tokens
                bi = buf.index(self.enc.ESCAPE)
                l = 0
                # copy the rest ignoring the padding
                while l < len(leftover_tokens):
                    if bi < len(buf):
                        if self.enc.is_tpos(buf[bi]):
                            bi += 1
                        elif buf[bi] in [self.enc.ESCAPE, self.enc.DIAMOND]:
                            buf[bi] = leftover_tokens[l]
                            bi += 1
                            l += 1
                        else:
                            raise DecodeError("Invalid padding in %s" % us.fn_txt)
                    else:
                        buf.append(leftover_tokens[l])
                        l += 1
                        bi += 1
                self.orig_withpos[us.fn_txt] = buf    # all tokens, including cut off from top/bottom
                us.scratch[us.fn_txt] = copy.copy(buf)
                us.orig2scratch[us.fn_txt] = list(range(len(us.scratch[us.fn_txt]) + 1))   # Initially 1:1, differs after edits
            us.body_tokens = list()
            us.fn_txt = ""
        elif us.state == MSG:
            self.commitmsg = self.enc.decode(us.msg_tokens).lstrip()
            us.msg_tokens = list()
        elif us.state == INS:
            if us.brewing_edit.tpos == 0:
                us.brewing_edit.error = "Invalid tpos at %i" % c
            if us.brewing_edit.shift == -1:
                us.brewing_edit.error = "Invalid shift at %i" % c
            if us.brewing_edit.fuzzy == -1 and len(us.brewing_edit.error) == 0:
                us.brewing_edit.error = "Fuzzy is still -1 at INS state"
            self.edits.append(us.brewing_edit)
            us.brewing_edit = Edit("", 0, -1, [], [])
        else:
            assert 0, us.state

    def untokenize_new_token(self, us: UntokenizeState, t: int, c: int):
        # To debug, uncomment this:
        # print("%s TOKEN[%i] = %i \"%s\"" % (us.state, c, t, self.enc.decode([t]).replace("\n", "\\n")))
        if us.state == WAIT:
            if t == self.enc.MSG:
                us.state = MSG
            elif t == self.enc.FILE:
                us.state = FILENAME
            elif t == self.enc.CHUNK:
                us.state = CHUNK
            else:
                raise DecodeError("Invalid token %i follows escape at %i" % (t, c))
            return
        if us.state == FILENAME:
            if t == self.enc.ESCAPE:
                self.untokenize_finish_state(us, c)
                us.state = FILENAME_DIAMONDS
            else:
                us.fn_tokens.append(t)
            return
        if us.state == FILENAME_DIAMONDS:
            while t == self.enc.DIAMOND:
                return
            us.state = CODE
        if us.state in [CODE, CODE_FINISHING]:
            if us.state == CODE and t == self.enc.ESCAPE:
                us.state = CODE_FINISHING
            elif us.state == CODE_FINISHING and self.enc.is_tpos(t):
                us.body_tokens.append(t)
                self.untokenize_finish_state(us, c)
                us.state = WAIT
                return
            us.body_tokens.append(t)
            return
        if us.state == MSG:
            if t == self.enc.ESCAPE:
                self.untokenize_finish_state(us, c)
                us.state = WAIT
                return
            elif t == self.enc.FILE:
                self.untokenize_finish_state(us, c)
                us.state = FILENAME
                return
            else:
                us.msg_tokens.append(t)
            return

        if us.state == CHUNK:
            if self.enc.is_tpos(t):
                us.brewing_edit.tpos = t
                us.brewing_edit.fn = self.tpos2fn(t)
                if us.brewing_edit.fn == "unknown":
                    us.brewing_edit.error = "unknown tpos"
                us.state = DEL
            else:
                raise DecodeError("In chunk state position token is expected at %i" % c)
            return
        if us.state == DEL:
            if self.enc.is_tpos(t):
                raise DecodeError("Del section cannot end with tpos at %i" % c)
            if t == self.enc.ESCAPE:
                self.untokenize_finish_state(us, c)
                us.state = SHIFT
            else:
                us.brewing_edit.todel.append(t)
                self.untokenize_locate_edit(us)
            return
        if us.state == SHIFT:
            tmp = self.enc.decode([t])
            indexin = OFFSET_CHARS
            if len(tmp) != 1 or tmp not in indexin:
                raise DecodeError("Invalid shift token at %i, decoded to '%s'" % (c, tmp))
            us.brewing_edit.shift = indexin.index(tmp)
            us.brewing_edit.real_delstart = -1
            us.brewing_edit.fuzzy = self.untokenize_locate_edit(us)
            self.untokenize_finish_state(us, c)
            us.state = INS
            return
        if us.state == INS:
            if self.enc.is_tpos(t):
                raise DecodeError("Ins section cannot have position token at %i" % c)
            if t == self.enc.CHUNK:
                self.untokenize_finish_state(us, c)
                us.state = CHUNK
                return
            us.brewing_edit.toins.append(t)
            return
        assert 0, us.state

    def untokenize(self, process_tokens: List[int], full_orig_tokens: Dict[str, List[int]]):
        us = self.untokenize_init(full_orig_tokens)
        for c, t in enumerate(process_tokens):
            if t==self.enc.EOT:
                us.eot = True
            if us.eot:
                break
            self.untokenize_new_token(us, t, c)
        self.untokenize_finish_state(us, c)
        return us

    def tpos2fn(self, tpos: int):
        for fn, fn_tokens in self.orig_withpos.items():
            if tpos in fn_tokens:
                return fn
        return "unknown"

    def _lookahead_ignoring_tpos(self, haystack: List[int], cursor: int, needle: List[int]) -> Tuple[bool, int]:
        if cursor < 0:
            return False, 0
        c = cursor
        i = 0
        while i < len(needle):
            if c >= len(haystack):
                return False, 0
            if self.enc.is_tpos(haystack[c]):
                c += 1
                continue
            if haystack[c] != needle[i]:
                return False, 0
            i += 1
            c += 1
        return True, c

    def untokenize_locate_edit(self, us: UntokenizeState) -> int:
        return self.edit_location_find(us, len(self.edits), us.brewing_edit)

    def edit_location_find(self, us: UntokenizeState, ie: int, e: Edit) -> int:
        fn = e.fn
        if e.error:
            return -1
        orig = self.orig_withpos[fn]
        try:
            orig_i = orig.index(e.tpos)
        except ValueError:
            e.error = "Cannot apply chunk %i, position token %s not found" % (ie, self.enc.decode([e.tpos]), fn)
            return -1
        try:
            tpos_scratch_idx = us.orig2scratch[fn].index(orig_i)
        except ValueError:
            e.error = "Cannot apply chunk %i, position token %s found at %i, but it's not in the scratch map" % (ie, self.enc.decode([e.tpos]), orig_i)
            return -1
        lf_skipped = 0
        candidates = []
        sub = 0
        scratch = us.scratch[fn]
        incomplete_todel = e.todel
        if len(incomplete_todel) <= 1 and e.shift == -1:
            return -1
        if e.real_delstart == -1:
            # search
            while 1:
                sof = (tpos_scratch_idx - sub == 0)
                if sof or scratch[tpos_scratch_idx - sub - 1] == self.enc.LF:
                    good, real_delends = self._lookahead_ignoring_tpos(scratch, tpos_scratch_idx - sub, e.todel)
                    score = 0
                    if good:
                        if e.shift != -1:
                            score = abs(lf_skipped - e.shift)
                        else:
                            score = 0
                        candidates.append( (score, tpos_scratch_idx - sub, real_delends) )
                    # print(
                    #     "ie=%i" % ie,
                    #     "sub=%i" % sub,
                    #     "lookahead '%s'" % self.enc.decode(e.todel).replace("\n", "\\n"),
                    #     "trying '%s'" % termcolor.colored(self.enc.decode(scratch[tpos_scratch_idx - sub : tpos_scratch_idx - sub + len(e.todel)]).replace("\n", "\\n"), "green"),
                    #     "good", termcolor.colored(good, "green" if good else "red"),
                    #     "score", score)
                    lf_skipped += 1
                if sof:
                    break
                if lf_skipped > TPOS_HIGH_WATER:
                    break
                sub += 1
            candidates.sort()
            if e.shift != -1 and len(candidates) == 0:
                e.error = "Cannot apply chunk %i, cannot find todel tokens %s + shift %i" % (ie, self.enc.decode([e.tpos]), e.shift)
                return -1
            if len(candidates) == 1 or (e.shift != -1 and len(candidates) > 0):
                fuzzy, e.real_delstart, e.real_delends = candidates[0]
                return fuzzy
            return -1
        else:
            # no need to search, confirm existing
            good, _ = self._lookahead_ignoring_tpos(scratch, e.real_delstart, e.todel)
            if not good:
                e.error = "Cannot apply chunk %i, cannot confirm todel tokens %s + shift %i" % (ie, self.enc.decode([e.tpos]), e.shift)
            else:
                e.real_delends = e.real_delstart + len(e.todel)
        return -1

    def edit_apply(self, us: UntokenizeState, ie: int, e: Edit):
        assert e.fn in us.scratch
        scratch = us.scratch[e.fn]
        orig2scratch = us.orig2scratch[e.fn]
        assert e.real_delstart != -1
        for future_edit in self.edits[ie+1:]:
            if future_edit.real_delstart == -1:
                continue
            if future_edit.fn != e.fn:
                continue
            if future_edit.error:
                continue
            if future_edit.real_delstart > e.real_delstart:
                shift = -(e.real_delends - e.real_delstart) + len(e.toins)
                future_edit.real_delstart += shift
                future_edit.real_delends += shift
        good, _ = self._lookahead_ignoring_tpos(scratch, e.real_delstart, e.todel)
        if not good:
            e.error = "cannot confirm todel tokens"
            return
        scratch[e.real_delstart:e.real_delends] = e.toins
        orig2scratch[e.real_delstart:e.real_delends] = [-1] * len(e.toins)
        us.stats["chunks_applied"] += 1

    def apply_edits_return_dest(self, us: UntokenizeState):
        fn_unchanged = set(fn for fn in us.scratch)
        fn_changed = set()
        self.errors.clear()
        for ie, e in enumerate(self.edits):
            # print("\napply %i" % ie)
            if e.shift == -1:
                # unfinished chunk, nothing we can do
                continue
            if e.error:
                self.errors.append(e.error)
                continue
            assert e.fuzzy != -1
            us.stats["fuzzy"] += e.fuzzy
            if e.fuzzy:
                print("chunk%i fuzzy" % ie, e.fuzzy)
            self.edit_apply(us, ie, e)
            if e.error:
                self.errors.append(e.error)
                continue
            fn_changed.add(e.fn)
        fn_unchanged -= fn_changed
        for fn, scratch in us.scratch.items():
            while len(scratch) and (scratch[-1] in [self.enc.ESCAPE, self.enc.DIAMOND] or self.enc.is_tpos(scratch[-1])):
                scratch.pop(-1)
            self.dest_tokens[fn] = [int(t) for t in scratch if not self.enc.is_tpos(t)]
        us.stats["errors"] = len(self.errors)
        us.stats["files_unchanged"] = len(fn_unchanged)
        us.stats["files_patched"] = len(fn_changed)
        return self.dest_tokens


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


# A longer example to test file cutting
example_odm = {
    "orig": {"courses/views.py": "from django.shortcuts import render\nfrom django.views.generic import ListView, DetailView, View\nfrom .models import Course, Lesson\n\n\nclass CourseListView(ListView):\n    model = Course\n\n\nclass CourseDetailView(DetailView):\n    model = Course\n\n\nclass LessonDetailView(View):\n\n    def get(self, request, course_slug, lesson_slug, *args, **kwargs):\n        course_qs = Course.objects.filter(slug=course_slug)\n        if course_qs.exists():\n            course = course_qs.first()\n        lesson_qs = course.lessons.filter(slug=lesson_slug)\n        if lesson_qs.exists():\n            lesson = lesson_qs.first()\n        context = {\n            'object': lesson\n        }\n\n        return render(request, \"courses/lesson_detail.html\", context)\n"},
    "commitmsg": "lisson view",
    "dest": {"courses/views.py": "from django.shortcuts import render, get_object_or_404\nfrom django.views.generic import ListView, DetailView, View\n\nfrom member_ships.models import UserMembership\nfrom .models import Course, Lesson\n\n\nclass CourseListView(ListView):\n    model = Course\n\n\nclass CourseDetailView(DetailView):\n    model = Course\n\n\nclass LessonDetailView(View):\n\n    def get(self, request, course_slug, lesson_slug, *args, **kwargs):\n        course_qs = Course.objects.filter(slug=course_slug)\n        lesson = None\n        course = None\n        if course_qs.exists():\n            course = course_qs.first()\n        lesson_qs = course.lessons.filter(slug=lesson_slug)\n        if lesson_qs.exists():\n            lesson = lesson_qs.first()\n        context = {\n            'object': None\n        }\n        # print(request.user)\n        # user_membership = UserMembership.objects.filter(user=request.user).first()\n        # user_membership_type = user_membership.membership.membership_type\n        # course_allowed_mem_types = course.allowed_membership.all()\n        user_membership = get_object_or_404(UserMembership, user=request.user)\n        print(\"user_member ship --------\")\n        print(user_membership)\n\n        user_membership_type = user_membership.membership.membership_type\n        course_allowed_mem_types = course.allowed_membership.all()\n\n        if course_allowed_mem_types.filter(membership_type=user_membership_type).exists():\n            context = {'object': lesson}\n        return render(request, \"courses/lesson_detail.html\", context)\n"}
}


def self_test(enc: SMCEncoding, odm: Dict[str, Any], verbose: bool, n_ctx: int, tight_shrink: bool=False):
    import time
    t0 = time.time()
    test1 = ContrastDiff(enc)
    full_orig_tokens = test1.from_odm_dict(odm, n_ctx,
        tight_shrink=tight_shrink,
    )
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
    enc = SMCEncoding("openai_programming_v2")
    self_test(enc, example_odm, verbose=True, n_ctx=2049)

