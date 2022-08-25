import re
from configs import Config_ru_en
from nltk.tokenize import word_tokenize

def parse_text(text, language):
    parsed_text = re.sub(Config_ru_en.lang_to_sub.get(language), ' ', text).strip()
    return parsed_text

def parse_text_kz(text):
    parsed_text = re.sub('([^А-Яа-яa-zA-ZӘәҒғҚқҢңӨөҰұҮүІі-]|[^ ]*[*][^ ]*)', ' ', text).strip()
    return parsed_text

def chunks_f(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def custom_sent_tokenize(text, language, max_len):
    sents = []
    chunk = ''
    words = word_tokenize(text, language=language)[::-1]
    while len(words):
        
        word = words.pop()

        if len(chunk + ' ' + word) > max_len:
            sents.append(chunk.strip())
            chunk = ''

        chunk += ' '
        chunk += word
        
    sents.append(chunk)
    return sents
