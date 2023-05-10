from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, ElementUnpackContext
from code_contrast.contrast_2023q2.el_file import FileElement
from typing import List, Tuple, Optional


STATE_DEL, STATE_LINE_N, STATE_INS = "DEL", "LINE_N", "INS"


class ChunkElement(Element):
    def __init__(self, orig_file: Optional[FileElement]):
        super().__init__("CHUNK")
        self.orig_file = orig_file
        self.dest_text: List[str] = []
        self.i0 = -1
        self.i1 = -1
        self.j0 = -1
        self.j1 = -1
        self.to_del: List[str] = []
        self.to_ins: List[str] = []
        self.fuzzy = -1
        self.error = ""
        self._decode_state = STATE_DEL
        # self._tok_CHUNK = -1
        self._ins_tokens: List[int] = []
        self._del_tokens: List[int] = []
        self._tok_LINE = -1
        self._chunk_line = -1
        self._line_tokens: List[int] = []

    def assign_from_diff(self, dest_text: List[str], i0, i1, j0, j1):
        assert self.orig_file
        self.dest_text = dest_text
        self.i0 = i0
        self.i1 = i1
        self.j0 = j0
        self.j1 = j1
        self.to_del = self.orig_file.file_lines[i0:i1]
        self.to_ins = dest_text[j0:j1]
        self.fuzzy = 0

    def pack_init(self, cx: ElementPackingContext) -> Tuple[List[int], List[int]]:
        assert self.orig_file
        t = cx.enc.encode("CHUNK\n")
        for line in range(self.i0, self.i1):
            line_t = cx.enc.encode(self.orig_file.file_lines[line])
            t.extend(line_t)
        t.extend([cx.enc.ESCAPE] + cx.enc.encode("LINE%04d\n" % (self.i0,)))
        for j in range(self.j0, self.j1):
            t.extend(cx.enc.encode(self.dest_text[j]))
        m = [1]*len(t)
        return t, m


    @classmethod
    def unpack_init(cls, cx: ElementUnpackContext, init_tokens: List[int]) -> Element:
        el = ChunkElement(None)
        def should_be_single_token(s):
            seq = cx.enc.encode(s)
            assert len(seq) == 1, "\"%s\" is not one token %s, first token is \"%s\"" % (s, seq, cx.enc.decode([seq[0]]).replace("\n", "\\n"))
            return seq[0]
        el._tok_LINE = should_be_single_token("LINE")
        el._state = STATE_DEL
        return el

    def _switch_state(self, cx, new_state):
        # print(" -- switch state %s -> %s" % (self._state, new_state))
        if self._state == STATE_LINE_N:
            tmp = cx.enc.decode(self._line_tokens)
            try:
                self._chunk_line = int(tmp)
            except ValueError:
                pass   # stays -1
            # print("LINE collected self._line_tokens \"%s\" -> _chunk_line %i" % (tmp.replace("\n", "\\n"), self._chunk_line))
            self._line_tokens = []
            # fills fuzzy correctly, even if we know the location already
            self._locate_this_chunk_in_file_above(cx, force=True)
            # self.i2 = self.i1 + self._del_str -- not needed really
        self._state = new_state

    def unpack_more_tokens(self, cx: ElementUnpackContext) -> bool:
        while len(cx.tokens) > 1:
            t0 = cx.tokens[0]
            t1 = cx.tokens[1]
            # print("chunk.unpack %5i \"%s\"" % (t0, cx.enc.decode([t0]).replace("\n", "\\n")))
            if cx.fmt.is_special_token(t0):
                if self._state == STATE_DEL and t1 == self._tok_LINE:
                    self._switch_state(cx, STATE_LINE_N)
                    del cx.tokens[:2]
                    continue
                else:
                    # print("special token, must be next element, chunk over")
                    return True
            if self._state == STATE_LINE_N:
                t1_txt = cx.enc.decode([t1])
                self._line_tokens.append(t0)
                if "\n" in t1_txt:
                    self._switch_state(cx, STATE_INS)
                del cx.tokens[0]
            elif self._state == STATE_INS:
                self._ins_tokens.append(cx.tokens.pop(0))
            elif self._state == STATE_DEL:
                self._del_tokens.append(cx.tokens.pop(0))
                self._locate_this_chunk_in_file_above(cx, force=False)
            else:
                assert 0, "unknown state %s" % self._state
        return False

    def unpack_finish(self, cx: ElementUnpackContext):
        to_del_str = self._del_str(cx)
        to_ins_str = self._ins_str(cx)
        self.to_del = to_del_str.splitlines(keepends=True)
        self.to_ins = to_ins_str.splitlines(keepends=True)

    def _del_str(self, cx):
        if len(self._del_tokens):
            to_del_str = cx.enc.decode(self._del_tokens)
            if not to_del_str.startswith("\n"):
                raise ValueError("there is no \\n in between CHUNK and deleted text")
        else:
            to_del_str = "\n"
        return to_del_str[1:]

    def _ins_str(self, cx):
        if len(self._ins_tokens):
            to_ins_str = cx.enc.decode(self._ins_tokens)
            if not to_ins_str.startswith("\n"):
                raise ValueError("there is no \\n in between LINE and inserted text")
        else:
            to_ins_str = "\n"
        return to_ins_str[1:]

    def _locate_this_chunk_in_file_above(self, cx: ElementUnpackContext, force: bool) -> bool:
        if not self.orig_file or force:
            lst: List[Tuple[FileElement, int, int]] = []
            to_del_str = self._del_str(cx)
            lst = cx.lookup_file(to_del_str, self._chunk_line)    # possible locations
            if len(lst) == 1:
                # print("found one match for todel")
                file, i0, fuzzy = lst[0]
                self.orig_file = file
                self.i0 = i0
                self.fuzzy = fuzzy
            # self.orig_file = file
            # self.i0 = i0
            # self.i1 = i1
        return False
