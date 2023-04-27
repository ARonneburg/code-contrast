import random
import time
import termcolor

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from code_contrast.contrast_2023q2.el_file import FileElement
from code_contrast.contrast_2023q2.element import Format2023q2, Element, ElementPackingContext, ElementUnpackContext
from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional

    # def unpack_init(self, cx: ElementUnpackContext):
    #     pass

    # def unpack_more_tokens(self, cx: ElementUnpackContext) -> bool:
    #     """
    #     This function must either:
    #      * Continously consume cx.tokens from the beginning, for example with cx.tokens.pop(0).
    #      * Wait until there is enough tokens for the complete element, del cx.tokes[0:N] to consume N at once.
    #     Or do both.
    #     Return False if more tokens are needed, True if the element cannot consume anymore, the unpacker
    #     should move to the next.
    #     """
    #     raise NotImplementedError()

    # def unpack_finish(self, cx: ElementUnpackContext):
    #     """
    #     Called after unpack_more_tokens() returns True, or the model hits max_tokens and cannot
    #     produce more tokens.
    #     """
    #     pass

class Unpacker:
    def __init__(self, fmt: Format2023q2, initial_elements: List[Element]):
        self.result = initial_elements[:]
        self.fmt = fmt
        self.enc = fmt.enc
        self.cx = ElementUnpackContext(
            fmt,
            lookup_file_by_tokens=self.lookup_file_by_tokens,
            lookup_file_by_line_number=self.lookup_file_by_line_number,
        )
        self._constructing: Optional[Element] = None
        self._start_tokens = List[int]

    def lookup_file_by_tokens(self, tokens: List[str]) -> FileElement:
        return None

    def lookup_file_by_line_number(self, line_number: int) -> FileElement:
        return None

    def feed_tokens(self, toks: List[str]):
        self.cx.tokens.extend(toks)
        while len(self.cx.tokens):
            if self._constructing is not None:
                # print("+1++++ ", self.cx.tokens)
                finished = self._constructing.unpack_more_tokens(self.cx)
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
                        self._constructing = self.fmt.element_classes[klass].unpack_init(self.cx, seq)
                        # print("hurray started", self._constructing)
                        self.cx.tokens = self.cx.tokens[l:]
                        break
            if self._constructing is None:
                break

    def finish(self):
        if self._constructing is not None:
            el = self._constructing
            el.unpack_finish(self.cx)
            self._constructing = None
            self.result.append(el)
