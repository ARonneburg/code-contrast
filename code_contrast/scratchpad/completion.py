from code_contrast import log
from code_contrast.encoding import Encoding
from code_contrast.pprint import hlprint
from code_contrast.scratchpad.base import ScratchpadBase

from typing import List


class ScratchpadCompletion(ScratchpadBase):
    def __init__(self, enc: Encoding, prompt, **kwargs):
        super().__init__(enc, **kwargs)
        self._tokens: List[int] = []
        self._prompt = prompt

    def new_token(self, m, b, logits, heads, logits_intrusion=dict()):
        a = super().new_token(m, b, logits, heads, logits_intrusion)
        ai = a.item()
        if ai == self.enc.EOT:
            self.finish_reason = "eot"
        if ai in self.stop_tokens:
            self.finish_reason = "stoptoken"
        self.needs_upload = True
        self.generated_tokens_n += 1
        self._tokens.append(ai)
        return a

    def prompt(self, T: int):
        # For facebook infill:
        #self._tokens = [2] + self.enc.encode(self.call["prompt"])
        assert len(self._tokens) == 0
        self._tokens = self.enc.encode(self._prompt)
        log("---------- prompt ----------")
        log(hlprint(self._tokens, self.enc))
        log("---------- /prompt ----------")
        if len(self._tokens) > T:
            return []
        return self._tokens

    def completion(self, final: bool):
        return {"text": self.enc.decode(self._tokens, skip_zeros=True, cut_at_eot=True)}
