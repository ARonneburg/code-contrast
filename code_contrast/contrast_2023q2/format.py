import termcolor
from code_contrast.encoding.smc_encoding import SMCEncoding
from typing import List, Dict, Tuple, Callable, Type

from code_contrast.contrast_2023q2.element import Format2023q2, Element, ElementPackingContext
from code_contrast.contrast_2023q2.el_msg import MsgElement



def format_2023q2_escape(enc: SMCEncoding) -> Format2023q2:
    fmt = Format2023q2(enc)
    fmt.element_start_seq = {
    "SYSTEM": [enc.ESCAPE, *enc.encode("SYSTEM")],
    "USER": [enc.ESCAPE, *enc.encode("USER")],
    "ASSISTANT": [enc.ESCAPE, *enc.encode("ASSISTANT")],
    }
    fmt.element_classes = {
    "SYSTEM": MsgElement,
    "USER": MsgElement,
    "ASSISTANT": MsgElement,
    }
    ESCAPE = enc.ESCAPE
    fmt.is_special_token = lambda t: t==ESCAPE
    return fmt


# def format_2023q2_escape(enc: SMCEncoding) -> Format2023q2:
#     fmt = Format2023q2(enc)
#     fmt.el_start_tokens = {
#     "USER": [enc.ISSUE_START],
#     "ASSISTANT": [enc.ISSUE_COMMENT],
#     }
#     return fmt

    # # A way to find a subclass of Element
    # @staticmethod
    # def activation_token_sequence(enc: SMCEncoding) -> List[int]:
    #     """
    #     Tokens which should create an instance of subclass of this class.
    #     """
    #     raise NotImplementedError()