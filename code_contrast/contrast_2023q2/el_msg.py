from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, element_classes
from typing import List, Tuple


class MsgElement(Element):
    def __init__(self, msg_role: str, msg_text: str):
        super().__init__("MSG")
        self.msg_role = msg_role
        self.msg_text = msg_text

    def pack_init(self, cx: ElementPackingContext) -> Tuple[List[int], List[int]]:
        toks = cx.enc.encode(self.msg_role + " " + self.msg_text + "\n")
        return toks, [1]*len(toks)


element_classes["MSG"] = MsgElement
