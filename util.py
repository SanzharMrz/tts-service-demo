import re
from nltk.tokenize import word_tokenize, sent_tokenize

from configs import Config_ru_en


def parse_text(text, language):
    parsed_text = re.sub(Config_ru_en.lang_to_sub.get(language), ' ', text).strip()
    return parsed_text


def parse_text_kz(text):
    parsed_text = re.sub('([^А-Яа-яa-zA-ZӘәҒғҚқҢңӨөҰұҮүІі!?.-]|[^ ]*[*][^ ]*)', ' ', text).strip()
    return parsed_text


def chunks_f(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def r(t):
    return t.replace(".", "").replace("?", "").replace("!", "").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")


def custom_sent_tokenize(text, language, max_len):
    sents = sent_tokenize(text)
    res = []
    for sent in sents:
        chunk = ''
        words = word_tokenize(sent, language=language)[::-1]
        words = [r(w).strip() for w in words if r(w).strip()]
        while len(words):
            word = words.pop()

            if len(chunk + ' ' + word) > max_len:
                res.append(chunk.strip())
                chunk = ''

            chunk += ' '
            chunk += word

        res.append(chunk.strip())

    return res
