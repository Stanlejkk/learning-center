"""Hidden checks for dl_06_bpe."""
import random

TEXT = ("the theme of the thesis: then the other theory thereby thwarted them. "
        "naïve café owners say 'ça va' — 東京 is big 🙂 and the end. ") * 3


def _ref_train(text, vocab_size):
    ids, merges = list(text.encode("utf-8")), {}
    for k in range(vocab_size - 256):
        stats = {}
        for p in zip(ids, ids[1:]):
            stats[p] = stats.get(p, 0) + 1
        if not stats:
            break
        pair = max(stats, key=stats.get)
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(256 + k)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        ids, merges[pair] = out, 256 + k
    return merges, ids


def check_get_stats_and_merge(ns):
    st = ns["get_stats"]([1, 2, 3, 1, 2, 2, 2])
    assert st == {(1, 2): 2, (2, 3): 1, (3, 1): 1, (2, 2): 2}, f"get_stats counts overlapping adjacent pairs; got {st}"
    assert ns["merge"]([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99], "merge replaces each occurrence of the pair"
    assert ns["merge"]([5, 5, 5, 5, 5], (5, 5), 7) == [7, 7, 5], "merge is left-to-right and non-overlapping"


def check_train_merges(ns):
    tok = ns["BasicTokenizer"]()
    tok.train(TEXT, 256 + 20)
    ref, _ = _ref_train(TEXT, 256 + 20)
    assert len(tok.merges) == 20, f"vocab_size 276 means 20 merges, got {len(tok.merges)}"
    assert list(tok.merges.items()) == list(ref.items()), \
        "merges differ from the reference (most frequent pair; ties -> pair occurring first; new ids 256, 257, ...)"
    assert tok.vocab[256] == bytes(list(ref)[0]), "vocab[256] should be the bytes of the first merged pair"


def check_roundtrip(ns):
    tok = ns["BasicTokenizer"]()
    tok.train(TEXT, 300)
    rng = random.Random(0)
    alphabet = list("abcdefghij thé東京🙂\n,.")
    samples = [TEXT, "", "x", "🙂🙂🙂", "the café 東京"] + ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40))) for _ in range(20)]
    for s in samples:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, f"decode(encode(s)) != s for {s!r}"
        assert all(0 <= i < 300 for i in ids), "ids must be < vocab_size"


def check_encode_matches_training(ns):
    tok = ns["BasicTokenizer"]()
    tok.train(TEXT, 256 + 40)
    _, ref_ids = _ref_train(TEXT, 256 + 40)
    ids = tok.encode(TEXT)
    assert len(ids) == len(ref_ids), f"encoding the training text should give {len(ref_ids)} tokens, got {len(ids)} (apply merges in the order learned)"
    assert len(ids) < len(TEXT.encode("utf-8")), "BPE should compress the training text"
