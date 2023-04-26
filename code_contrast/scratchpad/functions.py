from scratchpad_bigcode import ScratchpadBigCode
from prompts import comment_each_line


class ScratchpadCommentEachLine(ScratchpadBigCode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix, self.suffix, self.selection = comment_each_line(self.selection)
