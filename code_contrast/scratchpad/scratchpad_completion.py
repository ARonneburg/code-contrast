from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase

from typing import List


class ScratchpadCompletion(ScratchpadBase):
    def __init__(self, enc: SMCEncoding, prompt, **kwargs):
        super().__init__(enc, **kwargs)
        self._tokens: List[int] = []
        self._prompt = prompt

    def new_token(self, m, b, logits, heads, logits_intrusion=dict()):
        a = super().new_token(m, b, logits, heads, logits_intrusion)
        ai = a.item()
        self.needs_upload = True
        self.generated_tokens_n += 1
        self._tokens.append(ai)
        if ai==self.enc.EOT:
            self.finish_reason = "eot"
        if ai in self.stop_tokens:
            self.finish_reason = "stoptoken"
        if len(self._tokens) > 3:
            if self.stop_lf_lf and self._tokens[-1] == self.enc.LF and self._tokens[-2] == self.enc.LF:
                self.finish_reason = "ins-stop-lflf"
            if self.stop_lf_lf_lf:
                if self._tokens[-3] == self.enc.LF and self._tokens[-2] == self.enc.LF and self._tokens[-1] == self.enc.LF:
                    self.finish_reason = "ins-stop-lflflf"
                elif self._tokens[-2] == self.enc.LFLF and self._tokens[-1] == self.enc.LF:
                    self.finish_reason = "ins-stop-lflflf"
                elif self._tokens[-2] == self.enc.LFLF and self._tokens[-1] == self.enc.LFLF:
                    self.finish_reason = "ins-stop-lflflf"
        return a

    def prompt(self, T: int):
        # For facebook infill:
        #self._tokens = [2] + self.enc.encode(self.call["prompt"])
        assert len(self._tokens) == 0
        self._tokens = self.enc.encode(self._prompt)
        if len(self._tokens) > T:
            return []
        return self._tokens

    def completion(self, final: bool):
        return {"text": self.enc.decode(self._tokens, skip_zeros=True, cut_at_eot=True)}
