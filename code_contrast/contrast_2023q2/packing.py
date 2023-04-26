import random
import time
import termcolor

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import hlprint

from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, Format2023q2
from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


class Packer:
    def __init__(self, fmt: Format2023q2):
        self.fmt = fmt
        self.enc: SMCEncoding = fmt.enc
        self.r: List[int] = list()
        self.m: List[int] = list()
        self.plan: List[Element] = list()
        self.cx: Optional[ElementPackingContext] = None

    def add_to_plan(self, f: Element):
        self.plan.append(f)

    def pack_context(self,
        *,
        start_from_plan_n: int,
        mask_from_plan_n: int,
        limit_ctx_n: int,
        limit_aux_n: int,
        add_eot: bool
    ):
        cx = ElementPackingContext(self.fmt, limit_ctx_n, limit_aux_n)
        self.assign_random_line0_to_files()
        plan_toks: List[List[int]] = [list() for _ in range(len(self.plan))]
        plan_mask: List[List[int]] = [list() for _ in range(len(self.plan))]
        cx.filled_ctx_n = 2 if add_eot else 0   # two is ESCAPE, EOT
        cx.filled_aux_n = 0
        for i, el in enumerate(self.plan[start_from_plan_n:]):
            t, m = el.pack_init(cx)
            assert len(t) == len(m)
            t.insert(0, self.enc.ESCAPE)
            m.insert(0, 1)
            self.r.extend(t)
            self.m.extend(m if i >= mask_from_plan_n else [0]*len(m))
            cx.filled_ctx_n += len(t)
            plan_toks[i].extend(t)
            plan_mask[i].extend(m)
        if cx.filled_ctx_n > cx.limit_ctx_n:
            excess = cx.filled_ctx_n - cx.limit_ctx_n
            cx.limit_aux_n = max(0, cx.limit_aux_n - excess)
            print("WARNING: initial filled_ctx_n %d > limit_ctx_n %d. Reduced limit_aux_n to %d" % (cx.filled_ctx_n, cx.limit_ctx_n, cx.limit_aux_n))
        for aux in [1, 0]:
            while 1:
                any_still_expanding = False
                for i, el in enumerate(self.plan[start_from_plan_n:]):
                    # print("expand %i %s" % (i, el.el_type), "filled_ctx_n %d < %d" % (cx.filled_ctx_n, cx.limit_ctx_n),  "filled_aux_n %d < %d" % (cx.filled_aux_n, cx.limit_aux_n))
                    any_still_expanding |= el.pack_inflate(cx, aux)
                    # print(
                    #     " => total ctx %i aux %i," % (cx.filled_ctx_n, cx.filled_aux_n),
                    #     "projected ctx_n+aux_n %i\n" % (cx.filled_ctx_n + cx.filled_aux_n),
                    # )
                if not any_still_expanding:
                    break
        self.r, self.m = [], []
        for i, el in enumerate(self.plan[start_from_plan_n:]):
            self.r.extend(plan_toks[i])
            self.m.extend(plan_mask[i] if i >= mask_from_plan_n else [0]*len(plan_mask[i]))
            t, m = el.pack_finish(cx)
            assert len(t) == len(m)
            self.r.extend(t)
            self.m.extend(m if i >= mask_from_plan_n else [0]*len(m))

        # for i, (toks, msk) in enumerate(zip(plan_toks, plan_mask)):
        #     self.r.append(self.enc.ESCAPE)
        #     self.m.append(1 if i >= mask_from_plan_n else 0)
        #     self.r.extend(toks)
        #     self.m.extend(msk if i >= mask_from_plan_n else [0]*len(msk))
        if add_eot:
            self.r.extend([self.enc.ESCAPE, self.enc.EOT])
            self.m.extend([1, 1])
        # print("projected filled_ctx_n %d < limit %d" % (cx.filled_ctx_n, cx.limit_ctx_n))
        # print("projected filled_aux_n %d < limit %d" % (cx.filled_aux_n, cx.limit_aux_n))
        # print("projected filled_ctx_n+filled_aux_n = %d < %d" % (cx.filled_ctx_n + cx.filled_aux_n, limit_ctx_n + limit_aux_n))
        # print("                       real context = %d" % (len(self.r),))
        assert len(self.r) == len(self.m)
        assert len(self.r) <= cx.filled_ctx_n + cx.filled_aux_n, "Packed tokens %d, upper bound on number of tokens %d. May be an internal bug, maybe toks_count_LINE is not the max value possible." % (len(self.r), cx.filled_ctx_n + cx.filled_aux_n)
        self.cx = cx   # keep for debugging

    def assign_random_line0_to_files(self):
        MINLINE = 1000
        MAXLINE = 9000
        for f in self.plan:
            if f.el_type == "FILE":
                f.formal_line0 = random.randint(MINLINE, MAXLINE)

    def dump_r(self):
        return hlprint(self.enc, self.r, self.m)

    def __repr__(self) -> str:
        ret = ""
        x: Element
        for x in self.plan:
            ret += repr(x) + "\n"
        return ret