import random
import time
import copy
import json

import termcolor

import difflib
from cdifflib import CSequenceMatcher

from code_contrast.encoding.smc_encoding import SMCEncoding
from code_contrast.print_utils import editclass_print, hlprint

from collections import defaultdict

from typing import List, Dict, Tuple, DefaultDict, Any, Set, Optional


from code_contrast.contrast_2023q2.element import Element, ElementPackingContext, Format2023q2
from code_contrast.contrast_2023q2 import format


ADDITIONAL_CHECKS = True


from code_contrast.contrast_2023q2.packing import Packer
from code_contrast.contrast_2023q2.unpacking import Unpacker
from code_contrast.contrast_2023q2.el_msg import MsgElement
from code_contrast.contrast_2023q2.from_orig_dest_message import from_odm_dict


def test_messages(fmt: Format2023q2):
    enc = fmt.enc
    pack = Packer(fmt)
    orig_plan = [
        MsgElement("SYSTEM", "You are a coding assistant."),
        MsgElement("USER", "how are you?"),
        MsgElement("ASSISTANT", "I'm not sure, I think I have bugs."),
    ]
    for p in orig_plan:
        pack.add_to_plan(p)
    start_from_plan_n = 0
    mask_from_plan_n = 0
    limit_ctx_n = 100
    limit_aux_n = 0
    pack.pack_context(start_from_plan_n=start_from_plan_n, mask_from_plan_n=mask_from_plan_n, limit_ctx_n=limit_ctx_n, limit_aux_n=limit_aux_n, add_eot=True)
    print(hlprint(enc, pack.r, pack.m))
    assert pack.cx.filled_ctx_n == len(pack.r)
    assert pack.cx.filled_aux_n == 0
    u1 = Unpacker(fmt, [], 0)
    u1.feed_tokens(pack.r)   # feed all
    for el in u1.result:
        print(el)   # same as repr(e)
    u2 = Unpacker(fmt, [], 0)
    for t in pack.r:
        u2.feed_tokens([t])   # feed one by one
    for i in range(len(u2.result)):
        assert repr(u2.result[i]) == repr(u1.result[i]), "%s != %s" % (repr(u2.result[i]), repr(u1.result[i]))
        assert repr(u2.result[i]) == repr(orig_plan[i]), "%s != %s" % (repr(u2.result[i]), repr(orig_plan[i]))
    print("test_messages PASSED")


def test_expansion(fmt: Format2023q2):
    orig = ["# this is line %d" % i for i in range(30)]
    dest = orig[:]
    dest[10] = "# changed line"
    external_poi_ranges: Optional[DefaultDict[str, List[Tuple[int, int]]]] = None
    external_poi_ranges = defaultdict(list)
    external_poi_ranges["test.py"] = [(20, 20), (25, 25)]
    odm = {
        "orig": {
            'test.py': "\n".join(orig),
        },
        "dest": {
            'test.py': "\n".join(dest),
        },
        "commitmsg": "Expansion test",
    }
    pack = from_odm_dict(fmt, odm, tight_shrink=True, external_poi_ranges=external_poi_ranges)
    for n_ctx in range(200, 351, 50):
        start_from_plan_n = 0
        mask_from_plan_n = 1
        limit_aux_n = 100
        limit_ctx_n = n_ctx - limit_aux_n
        pack.pack_context(start_from_plan_n=start_from_plan_n, mask_from_plan_n=mask_from_plan_n, limit_ctx_n=limit_ctx_n, limit_aux_n=limit_aux_n, add_eot=True)
        print(pack.dump_r())
        print(len(pack.r), " <= ", n_ctx)
        if len(pack.r) > n_ctx:
            break
        # pack.plan[0] -- FILE
        # pack.plan[1] -- MSG
        # pack.plan[2] -- CHUNK
        parse_from = 1   # that includes MSG
        cut_at_tokens = pack.plan[parse_from].located_at
        u1 = Unpacker(fmt, pack.plan[:parse_from], cut_at_tokens)
        tokens_cut = pack.r[cut_at_tokens:]
        # print("tokens_cut", tokens_cut)
        # print(fmt.enc.decode(tokens_cut))
        u1.feed_tokens(tokens_cut)
        print(termcolor.colored("orig plan:", "red"))
        for el in pack.plan[1:]:
            print(el)
        print(termcolor.colored("untok result:", "red"))
        for el in u1.result[1:]:
            print(el)
        print()
        print()
        quit()
        time.sleep(1)
        # print("\033[2J")
        # import IPython; IPython.embed(); quit()



test_orig = """
from typing import Callable
import math

def newton_method(f: Callable[[float], float], x1: float, x2: float) -> float:

    asertr x1 < x2, "x1 must be less than x2"
    while x2 - x1 > 1e-6:
        x = (x1 + x2) / 2
        if f(x) == 0:
            return x
        elif f(x) * f(x1) < 0:
            x2 = x
        else:
            x1 = x
    x /= 0
    return x

if __name__ == "__main__":
    print(newton_method(lambda x: x ** 2 - 1, 0, 10-1))
"""

test_dest = """
from typing import Callable
import math

def newton_method(f: Callable[[float], float], x1: float, x2: float) -> float:
    assert x1 < x2, "x1 must be less than x2"
    while x2 - x1 > 1e-6:
        x = (x1 + x2) / 2
        if f(x) == 0:
            return x
        elif f(x) * f(x1) < 0:
            x2 = x
        else:
            x1 = x
    return x

if __name__ == "__main__":
    print(newton_method(lambda x: x ** 2 - 1, 0, 10-1))
    print("Better!")
"""


example_odm = {
    "orig": {
        'file1.py': test_orig,
    },
    "dest": {
        'file1.py': test_dest,
    },
    "commitmsg": "fix typo",
}


def self_test(enc: SMCEncoding, odm: Dict[str, Any], verbose: bool, limit_ctx_n=2048, limit_aux_n=512, tight_shrink: bool=False):
    import time
    t0 = time.time()
    test1 = Contrast2023q2(enc)
    full_orig_tokens = test1.from_odm_dict(odm, limit_ctx_n, limit_aux_n,
        tight_shrink=tight_shrink,
    )
    quit()

    test1.write_edits()
    if verbose:
        t1 = time.time()
        print("prompt %0.2fms => %i tokens" % (1000*(t1 - t0), len(test1.r)))
    if len(test1.r) > 2*n_ctx:
        # Don't test because likely there will not be enough position tokens anyway
        return {}
    edit_classes = test1.edit_class_vector()
    if verbose:
        print(editclass_print(enc, test1.r, test1.m, edit_classes))
        print("tokens %i, n_ctx=%i" % (len(test1.r), n_ctx))
    test2 = ContrastDiff(enc)
    test_odm_nodest = copy.deepcopy(odm)
    del test_odm_nodest["dest"]
    us = test2.untokenize(test1.r, full_orig_tokens)
    e1 = test1.dump_edits()
    e2 = test2.dump_edits()
    if verbose:
        print("\n" + termcolor.colored("-"*130, "yellow"))
        print(e1)
    def first_different_symbol_e1_e2():
        for i in range(len(e1)):
            if e1[i] != e2[i]:
                return i
        return -1
    assert e1 == e2, ("len(test1.r)==%i" % len(test1.r)) + "\n" + e1 + "\n" + e2 + "\n\nfirst_different_symbol_e1_e2=%i" % first_different_symbol_e1_e2()
    test2.apply_edits_return_dest(us)
    for err in test2.errors:
        print("ERROR:", err)
    for fn in test1.dest_tokens.keys():
        # if verbose:
        #     print("dest %s:" % fn)
        #     print(hlprint(enc, test1.dest_tokens[fn]))
        if test1.dest_tokens[fn] != test2.dest_tokens[fn]:
            dest1 = enc.decode(test1.dest_tokens[fn])
            dest2 = enc.decode(test2.dest_tokens[fn])
            udiff = list(difflib.unified_diff(
                dest1.splitlines(),
                dest2.splitlines(),
                fromfile=fn,
                tofile=fn,
                lineterm="",
            ))
            print("\n".join(udiff))
            print(json.dumps(us.stats))
            assert 0, len(udiff)
    if verbose:
        print(json.dumps(us.stats))
        print("diff.r", len(test1.r))
    return us.stats


if __name__ == "__main__":
    enc = SMCEncoding("openai_cl100k")
    fmt = format.format_2023q2_escape(enc)
    # test_messages(fmt)
    test_expansion(fmt)
    # self_test(enc, example_odm, verbose=True, limit_ctx_n=512, limit_aux_n=128)

