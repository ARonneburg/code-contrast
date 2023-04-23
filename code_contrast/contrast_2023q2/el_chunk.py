from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, element_classes
from code_contrast.contrast_2023q2.el_file import FileElement
from typing import List, Tuple


class ChunkElement(Element):
    def __init__(self, orig_file: FileElement):
        super().__init__("CHUNK")
        self.orig_file = orig_file
        self.dest_text: List[str] = []
        self.i0 = -1
        self.i1 = -1
        self.j0 = -1
        self.j1 = -1
        self.formal_line = -1
        self.shift = -1
        self.to_del: List[str] = []
        self.to_ins: List[str] = []
        self.fuzzy = -1
        self.error = ""

    def assign_from_diff(self, dest_text: List[str], i0, i1, j0, j1):
        self.dest_text = dest_text
        self.i0 = i0
        self.i1 = i1
        self.j0 = j0
        self.j1 = j1

    def pack_init(self, cx: ElementPackingContext) -> Tuple[List[int], List[int]]:
        t = cx.enc.encode("CHUNK\n")
        for line in range(self.i0, self.i1):
            line_t = cx.enc.encode(self.orig_file.file_lines[line])
            t.extend(line_t)
        t.extend([cx.enc.ESCAPE] + cx.enc.encode("LINE%04d\n" % self.formal_line))
        for j in range(self.j0, self.j1):
            t.extend(cx.enc.encode(self.dest_text[j]))
        m = [1]*len(t)
        return t, m
