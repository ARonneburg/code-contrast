import re
import torch as th


def temperature_top_k_top_p_filtering(logits, temperature=1, top_k=0, top_p=0, filter_value=-float('Inf')):
    assert logits.dim() == 1

    temperature = min(temperature, 1.0)
    temperature = max(temperature, 0.0)
    logits = logits / (temperature + 0.01)
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < th.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = th.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def simple_stoplist_cut(orig: str, dest: str, head: int, tail: int) -> str:
    expanded_head = orig.rfind("\n", 0, head) + 1
    result = []
    for idx, line in enumerate(dest[expanded_head:-tail].splitlines(keepends=True)):
        re_patterns = "|".join([
            r"copyright", r"copyleft", r"(C)", r"©", r"author", r"license",
            r'[\w.+-]+@[\w-]+\.[\w.-]+',  # email
        ])
        for _ in re.finditer(re_patterns, line.lower()):
            return "".join(result)
        result.append(line if idx > 0 else line[head-expanded_head:])
    return "".join(result)
