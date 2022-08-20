import os
import re
import json
import argparse
from time import time

import torch
import torchaudio

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from nltk import sent_tokenize


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class Config:
    file_to_lang = {
        'SCANNED - 4108_18411976.pdf': 'kz',
        'SCANNED - 4110_353652508.pdf': 'kz',
        'Optimzed - tanirbergenova_a.a.berikbaeva_m.a._inzhenerlіk_mehanika_-_2__334400008.pdf_enc6407430.pdf': 'kz',
        'Optimzed - sotsialnoe_gosudarstvo_teorija_metodologija_mehanizmy_93821869_pdf_enc9546740.pdf': 'ru',
        'Optimzed - book_w3vlpeg_235482945.pdf': 'ru',
        'Optimzed - book_664181550_138791945.pdf': 'en',
        'Optimzed - 5_999694706.pdf': 'ru',
        'Optimzed - 104_860776228.pdf': 'ru',
        'Not Optimized - filipov_etnichekie_protsessy_v_rossijskom_megapolise_i_juzhno__kazahstanskom_regio_PAApkoj.pdf': 'ru',
        'Not Optimized - book_762283876.pdf_enc4442172.pdf': 'kz',
        'Not Optimized - book_717820121_346006649.pdf': 'en',
        'Not Optimized - book_246240674.pdf_enc3941923.pdf': 'kz'
    }
    lang_to_sub = {'en': '[^A-Za-z]+',
                   'ru': '[^А-Яа-я]+'}
    speaker = {'ru': 'aidar_v2',
               'en': 'lj_16khz'}
    sample_rate = 16_000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = 1
    end = 50
    skip_pages = 0
    max_len = 130
    num_chunks = 6


def chunks_f(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_text(text, language):
    parsed_text = re.sub(Config.lang_to_sub.get(language), ' ', text).strip()[:Config.max_len]
    return parsed_text


def get_funcs(language):
    speaker = Config.speaker.get(language)
    print(speaker, language)
    model = None
    if language == 'en':
        model, symbols, _, _, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                         model='silero_tts',
                                                         language=language,
                                                         speaker=speaker)
        Config.symbols = symbols
        Config.apply_tts = apply_tts
    elif language == 'ru':
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)
    model.to(Config.device)
    return model


def get_audios(model, language, texts):
    audios = []
    if language == 'en':
        audios = Config.apply_tts(texts=texts,
                                  model=model,
                                  sample_rate=Config.sample_rate,
                                  symbols=Config.symbols,
                                  device=Config.device)
    elif language == 'ru':
        audios = [model.apply_tts(texts=[text],
                                  sample_rate=Config.sample_rate)[0] for text in texts]
    return audios


def parse(args):
    files = map(lambda f: os.path.join(args.folderpath, f), filter(lambda x: 'ipynb' not in x, os.listdir(args.folderpath)))
    language = args.lang
    head, tail = os.path.split(args.folderpath)
    folder_name = os.path.join(head, f'{tail}_parsed')

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_json = dict()

    model = get_funcs(language)
    tok_lang = 'russian' if language == 'ru' else 'english'

    for filename in files:
        print(filename)
        _, tail = os.path.split(filename)
        file_prefix, _ = tail.split('.')

        with open(filename, 'r+') as file:
            text = file.read()


        if not os.path.exists(folder_name):
            os.mkdir(folder_name)


        # cut it to sentences
        sentences = sent_tokenize(text, language=tok_lang)
        # clear sentences
        clear_sentences = list(map(lambda t: parse_text(t, language), sentences))
                    
        # create response
        sub_files = []
        audios = []
        for chunk_ in chunks_f(clear_sentences, 5):
            audios.extend(get_audios(model, language, chunk_))

        for sent_id, (sent, audio) in enumerate(zip(clear_sentences, audios)):
            response_filename = f'{file_prefix}_sentid_{sent_id}.wav'
            save_path = os.path.join(folder_name, response_filename)
            torchaudio.save(save_path, audio.unsqueeze(0), Config.sample_rate)
            sub_files.append(response_filename)


        folder_json[tail] = sub_files
    json_filename = os.path.join(folder_name, f'audio.json')
    with open(json_filename, 'w') as f:
        json.dump(folder_json, f)

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folderpath", default='/opt/demo_files_en', 
                                        help="Path to files, please avoid to use '/' at the end")
    parser.add_argument("--lang", 
                                default='ru',
                                choices=['ru', 'en'],
                                help='Choice you language to set model and tokenizers')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    parse(args)
