import torch as th

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase

from typing import List, Any, Dict, Set, Optional, Union


class ScratchpadBigCode(ScratchpadBase):
    def __init__(
        self,
        enc: SMCEncoding,
        intent: str,
        cursor_file: str,
        cursor0: int,
        cursor1: int,
        function: str,
        max_edits: int,
        sources: Dict[str, str],
        poi: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(enc, **kwargs)
        self._tokens: List[int] = []
        self.intent = intent
        self.cursor_file = cursor_file
        self.cursor0 = cursor0
        self.cursor1 = cursor1
        self.function = function
        self.max_edits = max_edits
        self.sources = sources
        self.prefix = ""
        self.suffix = ""

    def before_token_selection(self, m, **unused) -> Dict[str, Any]:
        return dict()

    def after_token_selection(
            self,
            m,
            chosen_token: th.Tensor,
            **unused
    ) -> Dict[str, Any]:
        # self.needs_upload = True
        self.generated_tokens_n += 1
        t = chosen_token.item()
        self._tokens.append(t)
        if chosen_token == self.enc.EOT:
            self.finish_reason = "eot"
        if not self.finish_reason:
            self._completion.append(t)
        if chosen_token in self.stop_tokens:
            self.finish_reason = "stoptoken"
        if len(self._tokens) > 3:
            if self.stop_lf_lf and self._tokens[-1] == self.enc.LF and self._tokens[-2] == self.enc.LF:
                self.finish_reason = "stop-lflf"
            if self.stop_lf_lf_lf:
                if self._tokens[-3] == self.enc.LF and self._tokens[-2] == self.enc.LF and self._tokens[-1] == self.enc.LF:
                    self.finish_reason = "stop-lflflf"
                elif self._tokens[-2] == self.enc.LFLF and self._tokens[-1] == self.enc.LF:
                    self.finish_reason = "stop-lflflf"
                elif self._tokens[-2] == self.enc.LFLF and self._tokens[-1] == self.enc.LFLF:
                    self.finish_reason = "stop-lflflf"
        return dict()

    def prompt(self, T: int):
        source = ""
        for fn, text in self.sources.items():
            if self.cursor_file == fn:
                source = text
        lines = source.splitlines()
        if len(lines)==0:
            lines.append("\n")
        if lines[-1] == "" or lines[-1][-1] != "\n":
            lines[-1] += "\n"
        join_back = "\n".join(lines)
        self.prefix = join_back[:self.cursor0]
        self.suffix = join_back[self.cursor1:]
        prompt: List[int] = []
        prompt.append(self.enc.PREFIX)
        prompt.extend(self.enc.encode(self.prefix))
        prompt.append(self.enc.SUFFIX)
        prompt.extend(self.enc.encode(self.suffix))
        prompt.append(self.enc.INFILL)
        print(self.enc.decode(prompt))
        # TODO: replace with file cutting
        max_prompt = T - self.max_tokens
        if len(prompt) > max_prompt:
            prompt = prompt[-max_prompt:]
        self._tokens = prompt[:]
        self._completion = []
        return prompt

    def completion(self, final: bool):
        result = {}
        completion_text = self.enc.decode(self._completion)
        result[self.cursor_file] = self.prefix + completion_text + self.suffix
        return result
