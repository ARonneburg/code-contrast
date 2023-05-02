import torch as th

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.scratchpad.scratchpad import ScratchpadBase
from code_contrast.scratchpad import bigcode_prompts, utils

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

    def _split_source_prefix_suffix_selection(self, only_full_lines: bool = True):
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
        if only_full_lines:
            self.cursor0, self.cursor1, self.selection = utils.full_line_selection(self.cursor0, self.cursor1, join_back)
        else:
            self.selection = ""
        self.prefix = join_back[:self.cursor0]
        self.suffix = join_back[self.cursor1:]

    def prompt_infill(self, T: int):
        self._split_source_prefix_suffix_selection(only_full_lines=False)
        prompt: List[int] = [
            self.enc.PREFIX,
            *self.enc.encode(self.prefix),
            self.enc.SUFFIX,
            *self.enc.encode(self.suffix),
            self.enc.INFILL,
        ]
        self.debuglog(self.enc.decode(prompt))
        # TODO: replace with file cutting
        max_prompt = T - self.max_tokens
        if len(prompt) > max_prompt:
            prompt = prompt[-max_prompt:]
        self._tokens = prompt[:]
        self._completion.clear()
        return prompt

    def prompt_comment_each_line(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.comment_each_line(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt_make_code_shorter(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.make_code_shorter(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt_explain_code_block(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.explain_code_block(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt_time_complexity(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.time_complexity(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt_fix_bug(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.fix_bug(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt_add_console_logs(self):
        self._split_source_prefix_suffix_selection()
        prompt_txt = bigcode_prompts.add_console_logs(self.selection)
        prompt: List[int] = self.enc.encode(prompt_txt)
        return prompt

    def prompt(self, T: int):
        if self.function == "infill":
            return self.prompt_infill(T)
        elif self.function.startswith("comment-each-line"):
            return self.prompt_comment_each_line()
        elif self.function.startswith("make-code-shorter"):
            return self.prompt_make_code_shorter()
        elif self.function.startswith("explain-code-block"):
            return self.prompt_explain_code_block()
        elif self.function.startswith("time-complexity"):
            return self.prompt_time_complexity()
        elif self.function.startswith('fix-bug'):
            return self.prompt_fix_bug()
        elif self.function.startswith('add-console-logs'):
            return self.prompt_add_console_logs()
        else:
            raise NotImplementedError

    @staticmethod
    def _postprocess(text: str) -> str:
        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('\n'):
            text = text[:-1]
        return text

    def completion(self, final: bool):
        result = {}
        completion_text = self.enc.decode(self._completion)
        lines = completion_text.splitlines()
        if lines:
            last_line = lines[-1]
            if last_line.startswith("----"):
                self.finish_reason = "prompt-endmark"

        if self.function.startswith('comment-each-line'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + completion_text + self.suffix
        elif self.function.startswith('make-code-shorter'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + completion_text + self.suffix
        elif self.function.startswith('explain-code-block'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + self.selection + completion_text + self.suffix
        elif self.function.startswith('time-complexity'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + self.selection + '\n' + completion_text + self.suffix
        elif self.function.startswith('fix-bug'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + completion_text + self.suffix
        elif self.function.startswith('add-console-logs'):
            completion_text = self._postprocess(completion_text)
            result[self.cursor_file] = self.prefix + completion_text + self.suffix
        else:
            result[self.cursor_file] = self.prefix + completion_text + self.suffix
        self.debuglog("SELECTION: \"%s\"" % self.selection.replace("\n", "\\n"))
        self.debuglog("COMPLETION: \"%s\"" % completion_text.replace("\n", "\\n"))
        return result
