import torch as th
import termcolor
import time

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from typing import Callable, Union, List, Set, Dict, Any, Optional

from code_contrast.scratchpad.utils import temperature_top_k_top_p_filtering


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

    def before_token_selection(self, m, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def select_tokens(
            self,
            logits: th.Tensor,
            tokens: th.Tensor,
            chosen_tokens: th.Tensor,
            *,
            temperatures: th.Tensor,
            logits_intrusion: Optional[List[Dict[int, float]]] = None,
            top_ps: Optional[List[float]] = None,
            top_ks: Optional[List[int]] = None,
            **unused
    ):
        DEBUGLOG_TOP3 = True
        if logits_intrusion:
            for idx, intr in enumerate(logits_intrusion):
                for t, add in intr.items():
                    if DEBUGLOG_TOP3:
                        self.debuglog("logit for %s is %0.3f, adding %0.3f" % (
                            hlprint(self.enc, [t]),
                            logits[idx, -1, t],
                            add))
                    logits[idx, :, t] += add

        if top_ps is not None and top_ks is not None:
            for b in range(logits.shape[0]):
                logits[b, -1] = temperature_top_k_top_p_filtering(
                    logits[b, -1], temperature=temperatures[b],
                    top_p=top_ps[b], top_k=top_ks[b]
                )
            probs = logits.softmax(dim=-1)
        else:
            probs = (logits / temperatures).squeeze(1).softmax(dim=-1)
        tokens.copy_(th.multinomial(probs, num_samples=1), non_blocking=True)
        chosen_tokens.copy_(tokens, non_blocking=True)

        if DEBUGLOG_TOP3:
            self._log_top3(token=tokens[0], probs=probs[0])
        return dict()

    def after_token_selection(self, m, **kwargs) -> Dict[str, Any]:
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

    def _log_top3(
            self,
            token: th.Tensor,
            probs: th.Tensor,
    ):
        def _format(t: str, color: str):
            return "\"%s\"" % termcolor.colored(t.replace("\n", "\\n").replace("\r", "\\r"), color)

        text = _format(self.enc.decode([token.item()]), "green").ljust(25)
        text += " <= "
        probs3, top3idx = map(lambda x: x.ravel().cpu().numpy(), probs.topk(4))
        for p, i in zip(probs3, top3idx):
            text += " %i %s" % (i, _format(self.enc.decode([i]), "yellow"))
            text += " %0.1f%%" % (100 * p)
        self.debuglog("top3: %s" % text)
