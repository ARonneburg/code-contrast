import torch as th

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase

from typing import List, Any, Dict


class ScratchpadCompletion(ScratchpadBase):
    def __init__(self, enc: SMCEncoding, prompt, **kwargs):
        super().__init__(enc, **kwargs)
        self._tokens: List[int] = []
        self._prompt = prompt

    def before_token_selection(self, m, **unused) -> Dict[str, Any]:
        return dict()

    def after_token_selection(
            self,
            m,
            chosen_token: th.Tensor,
            **unused
    ) -> Dict[str, Any]:
        self.needs_upload = True
        self.generated_tokens_n += 1
        self._tokens.append(chosen_token.item())
        if chosen_token == self.enc.EOT:
            self.finish_reason = "eot"
        if chosen_token in self.stop_tokens:
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
        return dict()

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
