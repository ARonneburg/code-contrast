import torch as th

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase

from typing import List, Any, Dict, Set, Optional, Union, Tuple


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
        self.__prefix, self.__suffix, self.__selection = "", "", ""

        source = [text for fn, text in self.sources.items() if fn == self.cursor_file] or [""]
        self._source = source[0]
        self._completion = []

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
        t_str = self.enc.decode([t])
        if self.stop_lf and t_str.startswith("\n"):
            self.finish_reason = "stop-lf"
        if self.stop_lf_lf and t_str.startswith("\n\n"):
            self.finish_reason = "stop-lflf"
        return dict()

    def _get_prefix_suffix_selection(self) -> Tuple[str, str, str]:
        source = self._source
        lines = source.splitlines()
        if len(lines) == 0:
            lines.append("\n")
        if lines[-1] == "" or lines[-1][-1] != "\n":
            lines[-1] += "\n"
        join_back = "\n".join(lines)

        prefix = join_back[:self.cursor0]
        suffix = join_back[self.cursor1:]
        selection = join_back[self.cursor0:self.cursor1]
        return prefix, suffix, selection

    @property
    def prefix(self):
        if not self.__prefix:
            self.__prefix, _, _ = self._get_prefix_suffix_selection()
        return self.__prefix

    @prefix.setter
    def prefix(self, value):
        self.__prefix = value

    @property
    def suffix(self):
        if not self.__suffix:
            _, self.__suffix, _ = self._get_prefix_suffix_selection()
        return self.__suffix

    @suffix.setter
    def suffix(self, value):
        self.__suffix = value

    @property
    def selection(self) -> str:
        if not isinstance(self.__selection, str):
            _, _, self.__selection = self._get_prefix_suffix_selection()
        return self.__selection

    @selection.setter
    def selection(self, value):
        self.__selection = value

    def prompt(self, T: int):
        prefix, suffix, selection = self.prefix, self.suffix, self.selection
        if selection:
            prefix += selection

        prompt: List[int] = [
            self.enc.PREFIX,
            *self.enc.encode(prefix),
            self.enc.SUFFIX,
            *self.enc.encode(suffix),
            self.enc.INFILL,
        ]

        print(self.enc.decode(prompt))
        # TODO: replace with file cutting
        max_prompt = T - self.max_tokens
        if len(prompt) > max_prompt:
            prompt = prompt[-max_prompt:]
        self._tokens = prompt[:]
        self._completion.clear()
        return prompt

    def completion(self, final: bool):
        result = {}
        completion_text = self.enc.decode(self._completion)
        result[self.cursor_file] = self.prefix + completion_text + self.suffix
        return result
