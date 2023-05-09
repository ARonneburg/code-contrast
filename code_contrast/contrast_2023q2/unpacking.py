import random
import time
import termcolor

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from code_contrast.contrast_2023q2.el_file import FileElement
from code_contrast.contrast_2023q2.element import Format2023q2, Element, ElementPackingContext, ElementUnpackContext
from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional, Type

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

    def lookup_file(self, todel: str, line_n: int) -> List[Tuple[FileElement, int, int]]:
        print("lookup_file", todel, line_n)
        return []

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
