import torch as th
import time
import termcolor

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from code_contrast.contrast import contrast
from code_contrast.scratchpad.scratchpad import ScratchpadBase


from typing import Dict, Optional, Any, List


class ScratchpadDiff(ScratchpadBase):
    def __init__(
        self,
        enc: SMCEncoding,
        intent: str,
        cursor_file: str,
        cursor0: int,
        cursor1: int,
        function: str,
        max_edits: int,
        sources: Dict[str, str],
        **kwargs
    ):
        super().__init__(enc, **kwargs)
        self.intent = intent
        self.cursor_file = cursor_file
        self.cursor0 = cursor0
        self.cursor1 = cursor1
        self.function = function
        self.max_edits = max_edits
        self.sources = sources
        self.state_before_first_tpos = True
        self.diff: contrast.ContrastDiff = None
        self.diff_out: Optional[contrast.ContrastDiff] = None
        self.diff_out_us: Optional[contrast.UntokenizeState] = None
        self.highlight = []
        self.highlight16 = []
        self.t_cursor0 = -1
        self.t_cursor1 = -1
        self.tpos_cursor0 = -1
        self.tpos_cursor1 = -1
        self.edits_uploaded = 0
        self.prompt_edits = 0
        self.cursorfile_tokens1 = None
        self.cursorfile_tokens2 = None
        self.cursorfile_map2to1 = None
        self.increase_logits = []
        self.no_stop_tokens_until = -1
        self.selected_newlines = -1
        self.P1 = 0.35
        self.P2 = 0.20
        self.JP1 = 0.20

    def set_model_thresholds(self, P1, P2, JP1, **more):
        self.P1 = P1
        self.P2 = P2
        self.JP1 = JP1
        super().set_model_thresholds(**more)

    def before_token_selection(
            self,
            m: Any,
            b: int,
            logit: th.Tensor,
            heads: List[th.Tensor],
            **unused
    ) -> Dict[str, Any]:
        if self.state_before_first_tpos:
            if self.function == "highlight":
                self.highlight_method4(m, b, logit, heads)
            self.state_before_first_tpos = False
        prev_token = self.diff.r[-1]
        suggest_tokens = []
        logits_intrusion: Dict[int, float] = dict()
        if prev_token == self.enc.CHUNK:
            for tpos in self.increase_logits:
                logits_intrusion[tpos] = +4.5
        if (
                self.diff_out_us is not None and
                self.diff_out_us.state == contrast.DEL and
                self.diff_out_us.brewing_edit.real_delstart != -1 and
                self.diff_out_us.brewing_edit.fn == self.cursor_file
        ):
            e = self.diff_out_us.brewing_edit
            scratch = self.diff_out_us.scratch[e.fn]
            if self.tpos_cursor1 != -1:
                tokens2 = self.cursorfile_tokens2
                assert all(tokens2[i] == scratch[i] for i in range(len(tokens2)))
                # print("todel:", termcolor.colored(self.enc.decode(scratch[e.real_delstart:e.real_delends]), "yellow"))
                print("suggest: [%s]" % termcolor.colored(self.enc.decode(scratch[e.real_delends:e.real_delends + 8]), "blue"))
                suggest_tokens = scratch[e.real_delends:e.real_delends + 8]
                beyond_selection = self.diff_out_us.brewing_edit.real_delends - self.t_cursor1
                if beyond_selection >= -1:
                    extra_newlines = len([t for t in scratch[self.t_cursor1:self.diff_out_us.brewing_edit.real_delends] if t == self.enc.LF])
                    if extra_newlines >= 0:
                        logits_intrusion[self.enc.ESCAPE] = 3.0 + 0.5 * extra_newlines
                # edit works like this: scratch[e.real_delstart:e.real_delends] = e.toins
        return dict(
            logits_intrusion=logits_intrusion,
            suggest_tokens=suggest_tokens,
        )

    def after_token_selection(
            self,
            m,
            chosen_token: th.Tensor,
            **unused
    ) -> Dict[str, Any]:
        self.diff.r.append(chosen_token.item())
        self.diff_out_catch_up()
        self.generated_tokens_n += 1
        return dict()

    def toplevel_fields(self):
        return {"highlight_tokens": self.highlight, "highlight_lines": self.highlight16}

    def completion(self, final: bool):
        if final and self.diff_out_us is not None:
            self.diff_out_catch_up()
            self.finalize()
            dest_tokens = self.diff_out.apply_edits_return_dest(self.diff_out_us)
            result = {}
            for fn in dest_tokens:
                result[fn] = self.enc.decode(self.diff_out.dest_tokens[fn])
            self.debuglog("ScratchpadDiff: finalized", self.diff_out_us.stats, self.finish_reason)
            return result
        elif final:
            self.debuglog("ScratchpadDiff: nothing useful available")
            return None
        else:
            return None

    def finalize(self):
        if self.diff_out_us is not None:
            self.diff_out.untokenize_finish_state(self.diff_out_us, self.diff_out_cursor)

    def diff_out_catch_up(self):
        if self.diff_out_us is None:
            return
        def finish(reason):
            self.finish_reason = reason
            self.diff_out.untokenize_finish_state(self.diff_out_us, self.diff_out_cursor)
        try:
            while self.diff_out_cursor < len(self.diff.r):
                t = self.diff.r[self.diff_out_cursor]
                if t==self.enc.EOT:
                    finish("eot")
                    break
                self.diff_out.untokenize_new_token(self.diff_out_us, t, self.diff_out_cursor)
                if self.diff_out_us.state == contrast.CHUNK and self.max_edits >= 0 and len(self.diff_out.edits) - self.prompt_edits >= self.max_edits:
                    finish("max-edits")
                    break
                if self.diff_out_cursor >= self.no_stop_tokens_until and self.diff_out_us.state == contrast.INS:
                    if t in self.stop_tokens:
                        finish("ins-stoptoken")
                        break
                    if self.stop_lf_lf and (self.diff.r[self.diff_out_cursor - 1], t) == (self.enc.LF, self.enc.LF):
                        finish("ins-stop-lflf")
                        break
                if self.diff_out_us.state in [contrast.DEL, contrast.SHIFT]:
                    # print("TEST epos=%i in %s\n\n" % (self.diff_out_us.e_tpos, self.increase_logits))
                    if len(self.increase_logits) > 0 and (self.diff_out_us.brewing_edit.tpos not in self.increase_logits):
                        finish("out-of-selection")
                        break
                self.diff_out_cursor += 1
        except contrast.DecodeError as e:
            self.debuglog("Exception in diff_out.untokenize_new_token: %s" % e)
            self.finish_reason = "diff-application-error"

    def prompt_infill(self, T):
        for fn, text in self.sources.items():
            if self.cursor_file == fn:
                cut_slash_n = text[self.cursor0:]
                slash_n_idx = cut_slash_n.find("\n")
                if slash_n_idx >= 0:
                    cut_slash_n = cut_slash_n[slash_n_idx+1:]
                self.odm["orig"][fn] = text[:self.cursor0] + self.enc.decode([self.enc.INFILL]) + cut_slash_n
                self.odm["dest"][fn] = text[:self.cursor0] + self.enc.decode([self.enc.DIAMOND]) + cut_slash_n
            else:
                self.odm["orig"][fn] = text
        self.orig_tokens = self.diff.from_odm_dict(
            self.odm,
            n_ctx=(T - self.max_tokens),
            tight_shrink=True,
            exact_cx_lines0=2,
            exact_cx_lines1=0,
            )
        self.diff.write_edits()
        assert len(self.diff.edits) == 1
        while len(self.diff.r) > 0:
            t = self.diff.r.pop()
            if t == self.enc.DIAMOND:
                break
        del3more = 3
        while len(self.diff.r) > 0 and self.diff.r[-1] not in [self.enc.LF] and del3more > 0:
            self.diff.r.pop()
            del3more -= 1

    def prompt_edit_chain(self, T):
        minrev = 10000
        for fn, text in self.sources.items():
            if ":" not in fn:
                continue
            if self.function != "edit-chain":
                continue
            fn, revstr = fn.split(":")
            if self.cursor_file != fn:
                continue
            revision = int(revstr)
            if revision < minrev:
                minrev = revision
            else:
                continue
            self.odm["orig"][fn] = text
            # self.debuglog("revision", revision)
            # self.debuglog("EDIT CHAIN BASE", text)
        for fn, text in self.sources.items():
            if ":" in fn:
                continue
            self.odm["dest"][fn] = text
            # self.debuglog("EDIT CHAIN DEST", text)
        self.orig_tokens = self.diff.from_odm_dict(
            self.odm,
            n_ctx=(T - self.max_tokens),
            tight_shrink=True,
            exact_cx_lines0=2,
            exact_cx_lines1=0,
            )
        self.diff.write_edits()
        assert self.diff.r[-1] == self.enc.EOT
        self.diff.r = self.diff.r[:-1]
        self.prompt_edits = len(self.diff.edits)

    def prompt_normal_diff(self, T):
        # Highlight also goes here
        for fn, text in self.sources.items():
            self.odm["orig"][fn] = text
            if self.cursor_file == fn:
                # make sure cursor01 is visible
                self.odm["dest"][fn] = text[:self.cursor0] + self.enc.decode([self.enc.DIAMOND]) + text[self.cursor1:]
            else:
                self.odm["dest"][fn] = text
        self.orig_tokens = self.diff.from_odm_dict(
            self.odm,
            n_ctx=(T - self.max_tokens),
            tight_shrink=True,
            exact_cx_lines0=2,
            exact_cx_lines1=0,
            )
        self.diff_out = contrast.ContrastDiff(self.enc)
        self.diff_out_us = self.diff_out.untokenize_init(self.orig_tokens)
        self.diff_out_cursor = 0
        if self.cursor0 != -1:
            self._find_selection_in_tokens()
        if self.function == "highlight":
            self.diff.write_esc_chunk()
            return
        if self.selected_newlines in [0, 1]:
            # selected single line or atcursor, write most of the chunk immediately
            self.max_edits = 1
            # tpos = self.cursorfile_tokens2[self.tpos_cursor0]
            # assert self.enc.is_tpos(tpos)
            # self.diff.r.append(tpos)
            self.diff.write_edits()
            assert len(self.diff.edits) == 1
            while len(self.diff.r) > 0:
                t = self.diff.r.pop()
                if t == self.enc.DIAMOND:
                    break
            while len(self.diff.r) > 0 and self.diff.r[-1] not in [self.enc.LF]:
                self.diff.r.pop()
        elif self.cursorfile_tokens2 is not None:
            # multi line selection, logits
            i = self.t_cursor0
            over = False
            while 1:
                t = self.cursorfile_tokens2[i]
                if self.enc.is_tpos(t):
                    self.debuglog("diff-selection increase logits", hlprint(self.enc, [t]))
                    self.increase_logits.append(t)
                    if over: break
                if i >= self.t_cursor1:
                    if len(self.increase_logits) > 0:
                        break
                    over = True
                if i >= len(self.cursorfile_tokens2):
                    break
                i += 1
            self.increase_logits.append(self.enc.EOT)
            self.diff.write_esc_chunk()
        else:
            self.diff.write_esc_chunk()

    def prompt(self, T):
        t0 = time.time()
        self.diff = contrast.ContrastDiff(self.enc)
        self.odm = {
            "orig": dict(),
            "commitmsg": self.intent,
            "dest": dict(),
        }
        # "^(highlight|infill|diff-anywhere|diff-atcursor|diff-selection|edit-chain)$"
        if self.function == "infill":
            self.prompt_infill(T)
        elif self.function == "edit-chain":
            self.prompt_edit_chain(T)
        else:
            self.prompt_normal_diff(T)
        if len(self.diff.r) >= T:
            self.debuglog("PACKING FAILED\n")
            return []
        self.no_stop_tokens_until = len(self.diff.r)
        if self.diff_out is None:
            self.diff_out = contrast.ContrastDiff(self.enc)
            self.diff_out_us = self.diff_out.untokenize_init(self.orig_tokens)
            self.diff_out_cursor = 0
            self.diff_out_catch_up()
            if self.cursor0 != -1:
                self._find_selection_in_tokens()
        return self.diff.r

    def _find_selection_in_tokens(self):
        assert self.cursor0 > -1 and self.cursor1 > -1, "cursor not set cursor0=%i cursor1=%i" % (self.cursor0, self.cursor1)
        if self.cursorfile_tokens1 is None:
            tokens1, tokens2, map2to1 = self._fn_create_map2to1(self.cursor_file)    # works fast ~1ms
            self.cursorfile_tokens1 = tokens1
            self.cursorfile_tokens2 = tokens2
            self.cursorfile_map2to1 = map2to1
            assert len(map2to1) == len(tokens2)
        self.t_cursor0, self.tpos_cursor0 = self._find_cursor_in_tokens(self.cursor0)   # works slow
        if self.cursor1 != self.cursor0:
            self.t_cursor1, self.tpos_cursor1 = self._find_cursor_in_tokens(self.cursor1)
        else:
            self.t_cursor1, self.tpos_cursor1 = self.t_cursor0, self.tpos_cursor0
        # self.debuglog(
        #     termcolor.colored(self.enc.decode(tokens2[:self.t_cursor0]), "yellow") +
        #     termcolor.colored("|", "green") +
        #     termcolor.colored(self.enc.decode(tokens2[self.t_cursor0:self.t_cursor1]), "red") +
        #     termcolor.colored("|", "green") +
        #     termcolor.colored(self.enc.decode(tokens2[self.t_cursor1:]), "yellow")
        #     )
        self.selected_newlines = len([t for t in self.cursorfile_tokens2[self.t_cursor0:self.t_cursor1] if t == self.enc.LF])

    def _find_cursor_in_tokens(self, cursor):
        tokens1, tokens2, map2to1 = self.cursorfile_tokens1, self.cursorfile_tokens2, self.cursorfile_map2to1
        left = 0
        right = len(tokens2) - 1
        while left <= right:
            mid = (left + right) // 2
            no_tpos = tokens1[:map2to1[mid]]
            shortage = map2to1[mid] > len(tokens1)
            if shortage > 0:
                no_tpos += [0]*shortage
            chars = self.enc.decode(no_tpos)
            if len(chars) < cursor:
                left = mid + 1
            elif len(chars) > cursor:
                right = mid - 1
            else:
                break
        if self.enc.is_tpos(tokens2[mid]):
            mid += 1
        c = result = mid
        while c < len(tokens2):
            if self.enc.is_tpos(tokens2[c]):
                return result, c
            c += 1
        self.debuglog("Cannot find cursor position in area covered by position tokens. This indicates a wrong way to cut the file top/bottom.")
        return result, 0

    def _fn_create_map2to1(self, fn):
        self.diff_out_catch_up()
        tokens1 = self.diff.orig_tokens[fn]      # equals to self.diff_out
        tokens2 = self.diff_out.orig_withpos[fn] # all tokens including cutted out top/bottom, with postion tokens in the middle
        # print(hlprint(self.enc, tokens2))
        i1 = 0
        map2to1 = []
        # At the end after escape, only diamonds and the last tpos are allowed:
        seen_escape = False
        for i, t in enumerate(tokens2):
            map2to1.append(i1)
            if self.enc.is_tpos(t):
                pass
            elif t == self.enc.ESCAPE:
                seen_escape = True
                i1 += 1
            elif t == self.enc.DIAMOND:
                i1 += 1
            elif not seen_escape:
                assert t == tokens1[i1]
                i1 += 1
            else:
                assert 0
        return tokens1, tokens2, map2to1

    def highlight_method4(self, m: Any, b, logit, heads):
        t0 = time.time()
        x_bte = heads["x_bte"][b:b+1]
        first_bt = th.zeros_like(x_bte[:, :, 0])
        first_bt[:, 0] = 1
        diffhlpoint_bt = th.zeros_like(x_bte[:, :, 0])
        diffhlpoint_bt[:, -1] = 1
        # ed_joint = m.highlight_forward(x_bte, first_bt, diffhlpoint_bt)
        # e2_logits = m.bidir_2logits(ed_joint)
        inside = m.highlight_forward(x_bte, first_bt, diffhlpoint_bt)
        ed_joint = x_bte + inside
        e2_logits = m.bidir_2logits(m.bidir_2logits_ln(ed_joint))

        pd_hl = th.distributions.categorical.Categorical(logits=e2_logits[0] / 1.0)
        probs = pd_hl.probs
        t1 = time.time()
        tokens1, tokens2, map2to1 = self.cursorfile_tokens1, self.cursorfile_tokens2, self.cursorfile_map2to1
        # tokens1 without position tokens
        # tokens2 with position tokens
        # both tokens1 and tokens2 are full, not cut at top/bottom
        start = self.diff.fn2tstart[self.cursor_file]   # index in r
        end = start + len(self.diff.fn2tind[self.cursor_file])
        cut0 = self.diff_out.fn2cut0[self.cursor_file]
        self.highlight = []
        self.highlight16 = []
        inside_yellow = False
        inside_purple = False
        starts16 = -1
        ends16 = -1
        def no_longer_16():
            nonlocal starts16, ends16
            if starts16 == -1:
                return
            tmp1 = self.enc.decode(tokens1[:starts16])
            tmp2 = self.enc.decode(tokens1[:ends16])
            self.highlight16.append((len(tmp1), len(tmp2), 0.15))
            starts16 = -1
            ends16 = -1
        for ti in range(start, end):
            if self.diff.r[ti] == self.enc.ESCAPE:
                break
            assert tokens2[ti - start + cut0] == self.diff.r[ti]
            if self.diff.r[ti] != self.enc.LF:
                continue
            prev_lf = ti - 1
            while prev_lf >= start and self.diff.r[prev_lf] != self.enc.LF:
                prev_lf -= 1
            p0 = float(probs[ti][0].item())
            p1 = float(probs[ti][1].item())
            p2 = float(probs[ti][2].item())
            tokens0pos = map2to1[prev_lf + 1 - start + cut0]
            tokens1pos = map2to1[ti - start + cut0]
            want16 = 0
            if p1 > self.P1:
                want16 = 0.3
                inside_yellow = True
            elif p2 > self.P2 and inside_yellow:
                want16 = 0.15
                inside_purple = True
            else:
                inside_yellow = False
                inside_purple = False
                no_longer_16()
            if want16 > 0:
                if starts16 == -1:
                    starts16 = tokens0pos
                ends16 = tokens1pos
            self.debuglog(
                "%-60s" % (self.enc.decode(self.diff.r[prev_lf+1:ti+1]).replace("\n", "\\n")),
                termcolor.colored(
                    " %0.1f%% %0.1f%% %0.1f%%" % (100*p0, 100*p1, 100*p2),
                    ("magenta" if inside_purple else "red") if inside_yellow else None,
                ),
            )
            if inside_yellow:
                for tj in range(prev_lf+1, ti+1):
                    jp0 = float(probs[tj][0].item())
                    jp1 = float(probs[tj][1].item())
                    jp2 = float(probs[tj][2].item())
                    self.debuglog(
                        termcolor.colored(
                            "  %-20s" % (self.enc.decode(self.diff.r[tj:tj+1]).replace("\n", "\\n")),
                            "blue"),
                        termcolor.colored(
                            " %0.1f%%" % (100*jp1,),
                            "yellow" if (jp1 > self.JP1) else None,
                        )
                    )
                    if self.enc.is_tpos(self.diff.r[tj]):
                        continue
                    if jp1 > self.JP1:
                        tokens1pos = map2to1[tj - start + cut0]
                        tokens2pos = map2to1[tj + 1 - start + cut0]
                        jtmp1 = self.enc.decode(tokens1[:tokens1pos])
                        jtmp2 = self.enc.decode(tokens1[:tokens2pos])
                        self.highlight.append((len(jtmp1), len(jtmp2), 0.95))
                        # self.highlight.extend(self.enc.decode(tokens1[:tokens1pos]))
        no_longer_16()
        t2 = time.time()
        self.debuglog("highlight_method4 calc %0.2fs tokens %0.2fs" % (t1-t0, t2-t1))
