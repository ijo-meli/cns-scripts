from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Callable


@dataclass
class DataAlpha:
    data: bytes
    alphabet: set[int]


Options = list[Iterable[int]]


def generate_options(target: DataAlpha, others: Iterable[DataAlpha]) -> Options:
    L = len(target.data)
    assert all(map(lambda p: len(p.data) >= L, others))
    xor = lambda A, B: bytes([a ^ b for a, b in zip(A, B)])
    xors = [xor(target.data, other.data) for other in others]

    result = []
    for i in range(L):
        options = set()
        for c in target.alphabet:
            ok = True
            for xored, other in zip(xors, others):
                cxor = xored[i] ^ c
                if cxor not in other.alphabet:
                    ok = False
                    break
            if ok: options.add(c)
        result.append(options)
    return result


def load_wordlist(
        filename: str,
        preprocessing: Callable[[str], str] | None = None,
        filter_condition: Callable[[str], bool] | None = None,
        sort_key: Callable[[str], int] | None = None
) -> Iterable[str]:
    words = []
    with open(filename, 'r') as file:
        for line in file:
            word = line.rstrip()
            if preprocessing: word = preprocessing(word)
            if filter_condition(word): words.append(word)
    words = list(set(words))  # remove duplicates
    if sort_key: words.sort(key=sort_key)
    return words


@dataclass
class WordTree:
    word: str
    children: list[WordTree]

    def preview(self, depth: int) -> list[str]:
        if depth <= 1 or not self.children: return [self.word]
        child_previews = map(lambda c: c.preview(depth - 1), self.children)
        child_previews = reduce(lambda a, b: a + b, child_previews)
        previews = map(lambda p: p[0] + p[1], itertools.product([self.word], child_previews))
        return list(previews)


def generate_word_tree(options: Options, wordlist: Iterable[str], delimiters: set[str]) -> WordTree | None:
    @functools.cache
    def _gwt(i: int) -> list[WordTree] | None:
        if i >= len(options): return []
        opt = options[i:]
        trees = []
        for word in wordlist:
            word_options = itertools.product([word], delimiters)
            word_options = map(lambda p: p[0] + p[1], word_options)

            for word_option in word_options:
                char_codes = map(ord, word_option)
                valid = all(map(lambda p: p[0] in p[1], zip(char_codes, opt)))
                if not valid: continue

                children = _gwt(i + len(word_option))
                if children is None: continue
                trees.append(WordTree(children=children, word=word_option))

        if not trees: return None
        return trees

    results = _gwt(0)
    if not results: return None
    return WordTree(children=results, word="")


def manual(tree: WordTree, preview_depth: int = 3):
    stack = [tree]
    while stack[-1].children:
        # options
        top = stack[-1]
        for i in range(len(top.children)):
            print(f"--- ({i}) ---")
            previews = top.children[i].preview(preview_depth)
            for preview in sorted(previews): print(preview)
            print("")  # newline
        # current string
        current = reduce(lambda a, b: a + b, map(lambda t: t.word, stack))
        print(f"current=\"{current}\"")
        # actions
        try:
            action = input("action=")
            if action.startswith("back"):
                if len(stack) > 1: stack.pop()
            elif action.startswith("depth"):
                preview_depth = int(action.split()[1])
            else:
                index = int(action)
                stack.append(top.children[index])
        except Exception:
            print("invalid action")
    print(reduce(lambda a, b: a + b, map(lambda t: t.word, stack)))


alphabet = set(b"abcdefghijklmnopqrstuvwxyz ?,.")

str_lower = lambda s: s.lower()
str_from_alpha = lambda a: (lambda s: all(map(lambda c: ord(c) in a, s)))
wordlist = load_wordlist(
    "wordlists/english.txt",
    preprocessing=str_lower,
    filter_condition=str_from_alpha(alphabet.difference(set(" .,?"))),
    sort_key=lambda s: len(s)
)

da1 = DataAlpha(data=b"\x0b\x1d\x17O\x16N\r\t\x02\x07\x16\x0e\x19\nY\0\b\x1d\x0b\x1cW\x19\x1c"
                     b"A\x05\x13\x13\nW\x06\x18\x13\x1b\x17\x16\x1b\x12", alphabet=alphabet)
da2 = DataAlpha(data=b"\x19\x1a\0O\x04\x01\f\r\x1eR\n\tW\x03\f\x02\x05R\x02\x1d\x12\x0f\r\x04\x1f"
                     b"R\x11\x07\x1e\0\x1e\x12M\x06\r\x0e\x19N\0\x0e\x18", alphabet=alphabet)
da3 = DataAlpha(data=b"\x04\x1cE\x1b\x1f\x0bY\x04\x03\x16E\x02\x0eN\x1d\x04\f\x06\rO\x04\x06\x18"
                     b"\r\x01R\x07\nW\x0fY\x12\f\x11\x17\x06\x11\x07\x1a\x04", alphabet=alphabet)
da4 = DataAlpha(data=b"\x19\x1dE\x1b\x1f\x0bY\x02\x1f\x13\x16\x07\x1e\0\x1eA\x1e\x1a\n\x1d\x12\x1d"
                     b"Y\t\x02\x1d\x13\n\x04N\r\t\x1f\x1d\x10\b\x1fN\x1b\r", alphabet=alphabet)
da5 = DataAlpha(data=b"\x0f\x1d\x1cO\x13\x0b\n\b\x1f\x1d\x10\x1c[N\x1f\0\0\x1b\t\x06\x12\x1d"
                     b"Y\x11\x1f\x17\x15\x0e\x05\x0b\x1dA\n\x13\x1cP", alphabet=alphabet)

opt = generate_options(da1, [da2, da3, da4])
tree = generate_word_tree(opt, wordlist, {" ", ". ", ", ", "? "})
manual(tree, 2)

pass
