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
        self.shift = -1
        self.to_del: List[str] = []
        self.to_ins: List[str] = []
        self.fuzzy = -1
        self.error = ""
        self._decode_state = STATE_DEL
        self._decode_tokens: List[int] = []
        # self._tok_CHUNK = -1
        self._tok_LINE = -1

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
        assert self.orig_file.formal_line0 > 0
        t = cx.enc.encode("CHUNK\n")
        for line in range(self.i0, self.i1):
            line_t = cx.enc.encode(self.orig_file.file_lines[line])
            t.extend(line_t)
        t.extend([cx.enc.ESCAPE] + cx.enc.encode("LINE%04d\n" % (self.orig_file.formal_line0 + self.i0)))
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
        # el._tok_CHUNK = should_be_single_token("CHUNK")
        el._tok_LINE = should_be_single_token("LINE")
        el._state = STATE_DEL
        return el

    def _switch_state(self, new_state):
        print(" -- switch state %s -> %s" % (self._state, new_state))


    def unpack_more_tokens(self, cx: ElementUnpackContext) -> bool:
        while len(cx.tokens) > 1:
            t0 = cx.tokens[0]
            t1 = cx.tokens[1]
            print("chunk.unpack %5i \"%s\"" % (t0, cx.enc.decode([t0]).replace("\n", "\\n")))
            if cx.fmt.is_special_token(t0):
                if self._state == STATE_DEL and t1 == self._tok_LINE:
                    self._switch_state(STATE_LINE_N)
                else:
                    print("special token, must be next element, chunk over")
                    return True
            self._decode_tokens.append(cx.tokens.pop(0))
            # cx.lookup_file_by_tokens
        return False

    # def unpack_finish(self, cx: ElementUnpackContext):
    #     t = cx.enc.decode(self._unpack_tokens)
    #     if t.startswith(" "):
    #         t = t[1:]
    #     if t.endswith("\n"):
    #         t = t[:-1]
    #     self.msg_text = t
