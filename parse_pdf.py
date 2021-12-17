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


def parse():
    files = filter(lambda x: 'ipynb' not in x, os.listdir('PDF_Examples/'))

    for filename in files:
        filename = 'PDF_Examples/Optimzed - book_664181550_138791945.pdf'
        head, tail = os.path.split(filename)
        folder_name, _ = tail.split('.')
        language = Config.file_to_lang.get(tail)
        model = get_funcs(language)

        book_json = {}
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        for page_num, page_layout in enumerate(extract_pages(filename, password="")):
            if page_num < Config.start:
                continue
            if page_num > Config.end:
                break
            if not page_num >= Config.skip_pages:
                continue

            page_height = page_layout.height
            chunks = [
                {
                    "audios": [],
                    "ybox": (
                    (Config.num_chunks - i - 1) / Config.num_chunks * page_height, (Config.num_chunks - i) / Config.num_chunks * page_height),
                }
                for i in range(Config.num_chunks)
            ]

            for current_chunk_num, chunk in enumerate(chunks):
                # collect raw text
                raw_page_text = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        for line in element:
                            chunk_num = int(((page_height - (line.bbox[1] + line.bbox[3]) / 2) / page_height) * Config.num_chunks)
                            if chunk_num != current_chunk_num:
                                continue
                            raw_text = line.get_text()
                            clear_text = parse_text(raw_text, language)
                            if len(clear_text) > 5:
                                raw_page_text.append(raw_text)

                # cut it to sentences
                tok_lang = 'russian' if language == 'ru' else 'english'
                sentences = list(filter(lambda x: len(x) > 10, sent_tokenize(' '.join(raw_page_text).replace('\n', ''), language=tok_lang)))
                # clear sentences
                clear_sentences = list(map(lambda t: parse_text(t, language), sentences))
                # generate audios
                audios = []
                for chunk_ in chunks_f(clear_sentences, 5):
                    audios.extend(get_audios(model, language, chunk_))

                # create response jsons
                parsing_results = []
                for sent_id, sent in enumerate(clear_sentences):
                    response_filename = f'page_{page_num}_chunk_num_{current_chunk_num}_sentid_{sent_id}.wav'
                    response_json = {
                        "page": page_num,
                        'chunk_num': current_chunk_num,
                        "sent_id": sent_id,
                        "text": sent,
                        "filename": response_filename
                    }
                    parsing_results.append(response_json)
                    chunk['audios'].append(response_filename)

                # create audio wav files
                for audio, json_ in zip(audios, parsing_results):
                    save_path = os.path.join(folder_name, json_['filename'])
                    torchaudio.save(save_path, audio.unsqueeze(0), Config.sample_rate)

            book_json[page_num] = {
                                        "height": page_height,
                                        "chunks": chunks
                                    }

            json_filename = os.path.join(folder_name, f'audio.json')
            with open(json_filename, 'w') as f:
                json.dump(book_json, f)
        break
