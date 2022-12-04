import torch as th
import termcolor
import time

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from typing import Callable, Union, List, Set


class ScratchpadBase:
    def __init__(
        self,
        enc: SMCEncoding,
        id: str,
        created: float,
        temperature: float,
        max_tokens: int,
        stop_tokens: Union[str, List[str]],
        logger: Callable,
        **unused,
    ):
        self.enc = enc
        self.id = id
        self._logger = logger
        self.created = created
        self.finish_reason = ""
        self.temp = min(max(float(temperature), 0.0), 1.0)
        self.max_tokens = int(max_tokens)
        tmp = stop_tokens
        if isinstance(tmp, str):
            stop_strings = [tmp]
        else:
            stop_strings = tmp
        self.stop_tokens: Set[int] = set()
        self.stop_lf_lf = False
        self.stop_lf_lf_lf = False
        for s in stop_strings:
            if s == "\n\n":
                self.stop_lf_lf = True
            if s == "\n\n\n":
                self.stop_lf_lf_lf = True
                continue
            t = self.enc.encode(s)
            if len(t) == 1:
                self.stop_tokens.add(t[0])
            else:
                self.debuglog("ScratchpadBase: cannot use '%s' as a stop token" % s)
        for k, v in unused.items():
            self.debuglog("ScratchpadBase: unused parameter '%s' = '%s'" % (k, v))
        self.generated_tokens_n = 0
        self.needs_upload = False

    def before_token_selection(self, **kwargs):
        raise NotImplementedError()

    def after_token_selection(self, **kwargs):
        raise NotImplementedError()

    def toplevel_fields(self):
        return {}

    def prompt(self, T: int):
        raise NotImplementedError()

    def completion(self, final: bool):
        raise NotImplementedError()

    def finalize(self):
        pass

    def set_model_thresholds(self, **args):
        if len(args) > 0:
            self.debuglog("set_model_thresholds: unused parameters %s" % str(args))

    def debuglog(self, *args):
        elapsed = time.time() - self.created
        self._logger("%4.0fms" % (elapsed * 1000,), *args)
