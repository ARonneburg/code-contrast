import code_contrast
from code_contrast.encoding.smc_encoding import SMCEncoding


def test_rev50000_derivatives(enc_name):
    print("\ntesting", enc_name)
    enc = SMCEncoding(enc_name)
    msg = "I can feel the magic, can you?\n\nПривет мир!!!"
    toks = enc.encode(msg)
    print("encode", toks)
    assert toks == [40, 460, 1254, 262, 5536, 11, 460, 345, 30, 198, 198, 140, 253, 21169, 18849, 38857, 16843, 20375, 12466, 120, 18849, 21169, 10185], toks
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

    long_text = open(code_contrast.encoding.smc_encoding.__file__).read()
    toks = enc.encode(long_text)
    print("big text", len(toks))
    print("lflf in toks", enc.LFLF in toks)
    # for i, tok in enumerate(toks):
    #     print("%03i %i \"%s\"" % (i, tok, enc.decode([tok]).replace("\n", "\\n")))
    assert long_text == enc.decode(toks)


if __name__ == "__main__":
    test_rev50000_derivatives("openai_reversible50000")
    test_rev50000_derivatives("openai_programming_v2")
