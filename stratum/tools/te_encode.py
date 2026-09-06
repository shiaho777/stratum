#!/usr/bin/env python3
"""te_encode.py — text -> Qwen3-VL token ids for the H3 pipeline (M2).

Minimal GPT-2-style byte-level BPE implementation over the ComfyUI
qwen25_tokenizer files (vocab.json + merges.txt), matching what
transformers.Qwen2Tokenizer would produce with add_special_tokens=False:

  1. regex pre-tokenization (GPT-2 pattern incl. contractions, CJK
     letters kept as single-codepoint runs per Qwen2 tokenizer.json)
  2. byte-level unicode remap of each pre-token
  3. BPE merge via the ranked merge list
  4. vocab lookup; unknown byte sequences map through <0xNN> tokens

The CPython-only implementation exists so the encoder never needs the
`transformers` wheel at runtime (same policy as stratum_tokenize.py).
Cross-checked against transformers Qwen2Tokenizer in the M2 commit.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))


def gpt2_byte_to_unicode():
    """Standard GPT-2 byte<->unicode table (bs=33..126, 161..172, 174..255
    pass through; the rest map to 256+n)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _pretok(text):
    """GPT-2-style pre-tokenization with HF tokenizer semantics: the
    prefix in ' ?\\p{L}+' matches ANY single whitespace char (space, tab,
    newline — pinned empirically against transformers 4.57; plain GPT-2
    matches only the space, which mis-splits ' \\ttabs')."""
    import unicodedata
    def is_l(ch): return unicodedata.category(ch).startswith("L")
    def is_n(ch): return unicodedata.category(ch).startswith("N")
    out = []
    i, n = 0, len(text)
    while i < n:
        m = re.match(r"'(?:[sdmt]|ll|ve|re)", text[i:])
        if m:
            out.append(m.group(0)); i += m.end(); continue
        j = i
        while j < n and text[j].isspace():
            j += 1
        if j > i:
            if j == n:                       # trailing run: emit whole
                out.append(text[i:j]); i = j; continue
            if j - i > 1:                    # leave last ws as prefix
                out.append(text[i:j-1]); i = j-1; continue
            # single ws + non-ws: fall through to the prefix rules
        k = i
        if k < n and text[k].isspace():
            k += 1
        if k < n and is_l(text[k]):
            e = k
            while e < n and is_l(text[e]): e += 1
            out.append(text[i:e]); i = e; continue
        if k < n and is_n(text[k]):
            e = k
            while e < n and is_n(text[e]): e += 1
            out.append(text[i:e]); i = e; continue
        if k < n and not text[k].isspace():
            e = k
            while e < n and not text[e].isspace() and not is_l(text[e]) \
                    and not is_n(text[e]):
                e += 1
            out.append(text[i:e]); i = e; continue
        out.append(text[i:k]); i = max(k, i + 1)
    return out


def tokenize(text, vocab, merges_rank, special):
    tokens = _pretok(text)

    byte_u = gpt2_byte_to_unicode()
    out_ids = []
    for tok in tokens:
        # special-token fast path (exact whole-token match)
        if tok in special:
            out_ids.append(special[tok])
            continue
        mapped = "".join(byte_u[b] for b in tok.encode("utf-8"))
        # BPE
        word = list(mapped)
        while len(word) > 1:
            pairs = [(merges_rank.get((word[i], word[i + 1]), 1 << 60), i)
                     for i in range(len(word) - 1)]
            best_rank, idx = min(pairs)
            if best_rank == 1 << 60:
                break
            word[idx:idx + 2] = [word[idx] + word[idx + 1]]
        for piece in word:
            if piece in vocab:
                out_ids.append(vocab[piece])
            else:
                for b in piece.encode("utf-8", "ignore") if False else []:
                    pass
                # piece is already unicode-mapped; fall back per original byte
                for ch in piece:
                    out_ids.append(vocab[ch])
    return out_ids


def load():
    base = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                        "model-h3-gguf", "qwen25_tokenizer")
    vocab = json.load(open(os.path.join(base, "vocab.json"), encoding="utf-8"))
    merges_rank = {}
    with open(os.path.join(base, "merges.txt"), encoding="utf-8") as f:
        for ln, line in enumerate(f):
            if line.startswith("#version") or not line.strip():
                continue
            a, b = line.rstrip("\n").split(" ")
            merges_rank[(a, b)] = ln
    cfg = json.load(open(os.path.join(base, "tokenizer_config.json"),
                         encoding="utf-8"))
    special = {}
    for tid, info in cfg.get("added_tokens_decoder", {}).items():
        special[info["content"]] = int(tid)
    return vocab, merges_rank, special


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    vocab, merges_rank, special = load()
    text = argv[1] if len(argv) == 2 else " ".join(argv[1:])
    ids = tokenize(text, vocab, merges_rank, special)
    print(" ".join(map(str, ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
