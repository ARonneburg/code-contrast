import torch as th

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase
from code_contrast.scratchpad import prompts

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
        # self.__prefix, self.__suffix, self.__selection = "", "", ""
        self._completion = []
        self.prefix: str = ""
        self.suffix: str = ""
        self.selection: str = ""


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

    def _split_source_prefix_suffix_selection(self):
        source = ""
        for fn, text in self.sources.items():
            if fn == self.cursor_file:
                source = text
                break
        lines = source.splitlines()
        if len(lines) == 0:
            lines.append("\n")
        if lines[-1] == "" or lines[-1][-1] != "\n":
            lines[-1] += "\n"
        join_back = "\n".join(lines)
        self.prefix = join_back[:self.cursor0]
        self.suffix = join_back[self.cursor1:]
        self.selection = join_back[self.cursor0:self.cursor1]

    def prompt_infill(self, T: int):
        self._split_source_prefix_suffix_selection()
        prompt: List[int] = []
        prompt.append(self.enc.PREFIX)
        prompt.extend(self.enc.encode(self.prefix))
        prompt.append(self.enc.SUFFIX)
        prompt.extend(self.enc.encode(self.suffix))
        prompt.append(self.enc.INFILL)
        prompt: List[int] = [
            self.enc.PREFIX,
            *self.enc.encode(self.prefix),
            self.enc.SUFFIX,
            *self.enc.encode(self.suffix),
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

    def prompt_with_prompt_engineering(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = prompts.comment_each_line(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt(self, T: int):
        if self.function == "infill":
            return self.prompt_infill(T)
        elif self.function.startswith("comment-each-line"):
            return self.prompt_with_prompt_engineering()
        else:
            raise NotImplementedError

    def completion(self, final: bool):
        result = {}
        completion_text = self.enc.decode(self._completion)
        lines = completion_text.splitlines()
        if lines:
            last_line = lines[-1]
            if last_line.startswith("---"):
                self.finish_reason = "prompt-endmark"
        result[self.cursor_file] = self.prefix + completion_text + self.suffix
        return result
