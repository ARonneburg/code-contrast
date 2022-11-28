import torch as th
import termcolor
import time
from typing import Callable, Union, List, Set

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from typing import Set


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

    def new_token(self, m, b, logits, heads, logits_intrusion=dict()):
        DEBUGLOG_TOP3 = False
        if self.temp <= 0.01:
            a = th.argmax(logits, dim=1)
        else:
            if logits_intrusion:
                for t, add in logits_intrusion.items():
                    if DEBUGLOG_TOP3:
                        self.debuglog("logit for %s is %0.3f, adding %0.3f" % (
                            hlprint(self.enc, [t]),
                            logits[-1, t],
                            add))
                    logits[-1, t] += add
            pd = th.distributions.categorical.Categorical(logits=logits / self.temp)
            a = pd.sample()
            if DEBUGLOG_TOP3:
                self._debug_top3(a, pd.probs)
        return a

    def _debug_top3(self, a: th.Tensor, probs: th.Tensor):
        def _format(t: str, color: str):
            return "\"%s\"" % termcolor.colored(t.replace("\n", "\\n").replace("\r", "\\r"), color)
        text = _format(self.enc.decode([a.item()]), "green").ljust(25)
        text += " <= "
        probs, top3idx = map(lambda x: x.ravel().cpu().numpy(), probs.topk(4, dim=1))
        for prob, i in zip(probs, top3idx):
            text += " %i %s" % (i, _format(self.enc.decode([i]), "yellow"))
            text += " %0.1f%%" % (100 * prob)
        self.debuglog("top3: %s" % text)

    def toplevel_fields(self):
        return {}

    def prompt(self, T: int):
        raise NotImplementedError()

    def completion(self, final: bool):
        raise NotImplementedError()

    def finalize(self):
        pass

    def debuglog(self, *args):
        elapsed = time.time() - self.created
        self._logger("%4.0fms" % (elapsed * 1000,), *args)
