import bpe_encoding


def test_rev50000_derivatives(ename):
    print("\ntesting", ename)
    enc = bpe_encoding.Encoding(ename)
    msg = "I can feel the magic, can you?\nПривет мир!!!"
    toks = enc.encode(msg)
    print("encode", toks)
    assert toks == [40, 460, 1254, 262, 5536, 11, 460, 345, 30, 198, 140, 253, 21169, 18849, 38857, 16843, 20375, 12466,
                    120, 18849, 21169, 10185], toks
    # Compare to https://beta.openai.com/tokenizer?view=bpe
    msg2 = enc.decode(toks)
    print("decode", msg2)
    assert msg2 == msg

    msg = "Hello world 1<|endoftext|>Hello world 2"
    toks = enc.encode(msg)
    assert toks == [15496, 995, 352, 27, 91, 437, 1659, 5239, 91, 29, 15496, 995, 362]
    assert enc.EOT == 50256
    toks.append(enc.EOT)
    print("encode", toks)
    msg2 = enc.decode(toks)
    print("decode", msg2)
    assert msg2 == msg + "<|endoftext|>"


def test_incoder_derivatives(fn):
    print("\ntesting", fn)
    enc = bpe_encoding.Encoding(fn)
    msg = "I can feel the magic, can you?\nПривет мир!!!:\n    if test"
    toks = enc.encode(msg)
    # toks = [0, 1, 2, 3, *toks]   # see special tokens
    # for i, t in enumerate(toks):
    #     print("%-4i %-4i \"%s\"" % (i, t, enc.decode([t]).replace("\n", "\\n")))
    assert toks == [11352, 39165, 1831, 12237, 267, 1220, 4973, 37, 205, 42554, 43564, 13560, 22569, 48832, 4942, 5336, 44014, 32, 205, 996, 695], toks
    msg2 = enc.decode(toks)
    assert msg2 == msg, msg2


def test_cpp_encoder(ename):
    enc = bpe_encoding.Encoding(ename)
    enc.load_cpp()
    msg = "I can feel the magic, can you?\nПривет мир!!!"
    tokens1 = enc.encode(msg)
    tokens2 = enc.senc.bpe(msg.encode("utf-8"))
    msg2 = enc.decode(tokens2)
    print("decode", msg2)
    assert tokens1 == tokens2


def test_stochastic_prob(ename, verbose=False):
    prob = 0.05
    print("\ntest_stochastic_prob", ename, prob)
    enc = bpe_encoding.Encoding(ename)
    msg = open(__file__).read()
    tokens1, _ = enc.encode_stochastic(msg, [], prob=prob)
    tokens2 = enc.encode(msg)
    if verbose:
        for t1, t2 in zip(tokens1, tokens2):
            print("%05i %05i  %20s %20s" % (t1, t2, enc.decode([t1]).replace("\n", "\\n"), enc.decode([t2]).replace("\n", "\\n")))
    print("len(tokens1)=%i len(tokens2)=%i   percent=%0.1f%%" % (len(tokens1), len(tokens2), 100*(len(tokens1)-len(tokens2))/len(tokens2)))


if __name__=="__main__":
    test_rev50000_derivatives("openai_reversible50000")
    test_rev50000_derivatives("openai_programming_v1")
    test_rev50000_derivatives("openai_programming_v2")
    test_cpp_encoder("openai_reversible50000")
    test_stochastic_prob("openai_programming_v1", verbose=False)
    test_incoder_derivatives("facebook_incoder")
    test_incoder_derivatives("fb1")
    test_incoder_derivatives("fb3")
