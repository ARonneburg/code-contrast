import torch as th
import termcolor

from code_contrast.encoding import Encoding
from code_contrast.pprint import log
from code_contrast.pprint import hlprint

from typing import Set


class ScratchpadBase:
    def __init__(self,
                 enc: Encoding,
                 id: str,
                 temperature,
                 max_tokens,
                 stop_tokens,
                 **unused):
        self.enc = enc
        self.id = id
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
        for s in stop_strings:
            if s == "\n\n":
                self.stop_lf_lf = True
                continue
            t = self.enc.encode(s)
            if len(t) == 1:
                self.stop_tokens.add(t[0])
            else:
                log("ScratchpadBase: cannot use '%s' as a stop token" % s)
        for k, v in unused.items():
            log("ScratchpadBase: unused parameter '%s' = '%s'" % (k, v))
        self.generated_tokens_n = 0
        self.needs_upload = False

    def new_token(self, m, b, logits, heads, logits_intrusion=dict()):
        if self.temp <= 0.01:
            a = th.argmax(logits, dim=1)
        else:
            if logits_intrusion:
                for t, add in logits_intrusion.items():
                    log("logit for %s is %0.3f, adding %0.3f" % (
                        hlprint([t], self.enc),
                        logits[-1, t],
                        add))
                    logits[-1, t] += add
            pd = th.distributions.categorical.Categorical(logits=logits / self.temp)
            probs = pd.probs
            top3idx = probs.topk(4, dim=1)[1]
            explain = ""
            a = pd.sample()
            explain += "\"%s\"" % (termcolor.colored(self.enc.decode([a.item()]).replace("\n", "\\n").replace("\r", "\\r"), "green"),)
            while len(explain) < 25:
                explain += " "
            explain += " <= "
            for i in top3idx[0]:
                i = i.item()
                explain += " %i \"%s\"" % (i, termcolor.colored(self.enc.decode([i]).replace("\n", "\\n").replace("\r", "\\r"), "yellow"),)
                explain += " %0.1f%%" % (100*probs[0, i].item())
            log("top3: %s" % explain)
        return a

    def toplevel_fields(self):
        return {}

    def prompt(self, T: int):
        raise NotImplementedError()

    def completion(self, final: bool):
        raise NotImplementedError()

    def finalize(self):
        pass
