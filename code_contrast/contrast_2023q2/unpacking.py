import random
import time
import termcolor

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from code_contrast.contrast_2023q2.el_file import FileElement
from code_contrast.contrast_2023q2.element import Format2023q2, Element, ElementPackingContext, ElementUnpackContext
from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional, Type


class Unpacker:
    def __init__(self, fmt: Format2023q2, initial_elements: List[Element], position: int):
        self.result = initial_elements[:]
        self.fmt = fmt
        self.enc = fmt.enc
        self.cx = ElementUnpackContext(
            fmt,
            lookup_file=self.lookup_file,
        )
        self._constructing: Optional[Element] = None
        self._position = position

    def lookup_file(self, todel: str, external_line_n: int) -> List[Tuple[FileElement, int, int]]:
        print("lookup_file \"%s\" external_line_n=%i" % (todel.replace("\n", "\\n"), external_line_n))
        if len(todel) == 0:
            return []
        lst = []
        for potential_file in self.result:
            if potential_file.el_type == "FILE":
                file: FileElement = potential_file
                cursor = 0
                for _ in range(5):  # pointless to return more than 5
                    i = file._file_lookup_helper_string.find(todel, cursor)
                    if i == -1:
                        break
                    line_n = file._file_lookup_helper_string.count("\n", 0, i)
                    fuzzy = abs(external_line_n - line_n) if external_line_n != -1 else -1
                    lst.append((file, line_n, fuzzy))
                    cursor = i + 1
        return lst

    def feed_tokens(self, toks: List[int]):
        self.cx.tokens.extend(toks)
        while len(self.cx.tokens):
            if self._constructing is not None:
                # print("+1++++ ", self.cx.tokens)
                toks_before = len(self.cx.tokens)
                finished = self._constructing.unpack_more_tokens(self.cx)
                toks_after = len(self.cx.tokens)
                assert toks_after <= toks_before
                self._position += toks_before - toks_after
                # print("+2++++ ", self.cx.tokens)
                if finished:
                    el = self._constructing
                    el.unpack_finish(self.cx)
                    self._constructing = None
                    self.result.append(el)
                else:
                    # print("over")
                    break
            if self._constructing is None:
                for klass, seq in self.fmt.element_start_seq.items():
                    l = len(seq)
                    # print("does %s start with %s?" % (self.cx.tokens, seq))
                    if self.cx.tokens[:l] == seq:
                        # print("starting with", self.cx.tokens, " -> ", klass)
                        Class: Type[Element] = self.fmt.element_classes[klass]
                        self._constructing = Class.unpack_init(self.cx, seq)
                        self._constructing.located_at = self._position
                        # print("hurray started", self._constructing)
                        del self.cx.tokens[:l]
                        self._position += l
                        break
            if self._constructing is None:
                # print("cannot start", self.cx.tokens)
                break

    def finish(self):
        if self._constructing is not None:
            el = self._constructing
            el.unpack_finish(self.cx)
            self._constructing = None
            self.result.append(el)
