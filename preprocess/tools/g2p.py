import re

import ToJyutping
from g2pM import G2pM
from g2p_en import G2p as G2pE

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_WORD_RE = re.compile(r"[\u4e00-\u9fff]")
_JA_PHONEME_RE = re.compile(r"-(.+?)\+")

EN_FLAG = "en_"
YUE_FLAG = "yue_"
ZH_FLAG = "zh_"
JA_FLAG = "ja_"

g2p_zh = G2pM()
g2p_en = G2pE()

_pyopenjtalk = None


def _get_pyopenjtalk():
    global _pyopenjtalk
    if _pyopenjtalk is None:
        import pyopenjtalk  # pyopenjtalk-plus (API互換)
        _pyopenjtalk = pyopenjtalk
    return _pyopenjtalk


def is_chinese_char(word: str) -> bool:
    if len(word) != 1:
        return False
    return bool(_ZH_WORD_RE.fullmatch(word))

def is_english_word(word: str) -> bool:
    if not word:
        return False
    return bool(_EN_WORD_RE.fullmatch(word))

def g2p_japanese(word):
    """Convert a Japanese word to phonemes using pyopenjtalk fullcontext labels.

    Returns a string like ``ja_s-a-k-u-r-a`` (prefix only on the first phoneme)
    so that ``data_processor.py`` can split on ``-`` and re-apply the ``ja_``
    prefix to each element, matching the English phoneme convention.
    """
    labels = _get_pyopenjtalk().extract_fullcontext(word)
    phonemes = []
    for label in labels:
        m = _JA_PHONEME_RE.search(label)
        if m:
            ph = m.group(1)
            if ph in ("sil", "pau"):
                continue
            phonemes.append(ph)
    if not phonemes:
        return "<SP>"
    return JA_FLAG + "-".join(phonemes)


def g2p_cantonese(sent):
    return ToJyutping.get_jyutping_list(sent)       # with tone

def g2p_mandarin(sent):
    return g2p_zh(sent, tone=True, char_split=False)

def g2p_english(word):
    return g2p_en(word)

def g2p_transform(words, lang):

    if lang == "Japanese":
        transformed_words = []
        for w in words:
            if w == "<SP>":
                transformed_words.append("<SP>")
            else:
                w = w.replace("?", "").replace(".", "").replace("!", "").replace(",", "")
                transformed_words.append(g2p_japanese(w))
        return transformed_words

    zh_words = []
    transformed_words = [0] * len(words)

    for idx, w in enumerate(words):
        if w == "<SP>":
            transformed_words[idx] = w
            continue

        w = w.replace("?", "").replace(".", "").replace("!", "").replace(",", "")

        if is_chinese_char(w):
            zh_words.append([idx, w])
        else:
            if is_english_word(w):
                w = EN_FLAG + "-".join(g2p_english(w.lower()))
            else:
                w = "<SP>"
        transformed_words[idx] = w

    sent = "".join([k[1] for k in zh_words])

    # zh (zh and yue) transformer to g2p
    if len(sent) > 0:
        if lang == "Cantonese":
            g2pm_rst = g2p_cantonese(sent)       # with tone
            g2pm_rst = [YUE_FLAG + k[1] for k in g2pm_rst]
        else:
            g2pm_rst = g2p_mandarin(sent)
            g2pm_rst = [ZH_FLAG + k for k in g2pm_rst]
        for p, w in zip([k[0] for k in zh_words], g2pm_rst):
            transformed_words[p] = w

    return transformed_words

