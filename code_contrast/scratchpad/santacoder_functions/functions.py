from code_contrast.scratchpad.scratchpad_bigcode import ScratchpadBigCode
from prompts import comment_each_line


class ScratchpadCommentEachLine(ScratchpadBigCode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        arr = comment_each_line(self.selection)
        self.prefix, self.suffix, self.selection = arr.prefix, arr.suffix, arr.selection

