import os
import re
import json
import argparse

import torch
import torchaudio

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


class Config:
    file_to_lang = {'SCANNED - 4108_18411976.pdf': 'kz',
                    'SCANNED - 4110_353652508.pdf': 'kz',
                    'Optimzed - tanirbergenova_a.a.berikbaeva_m.a._inzhenerlіk_mehanika_-_2__334400008.pdf_enc6407430.pdf': 'kz',
                    'Optimzed - sotsialnoe_gosudarstvo_teorija_metodologija_mehanizmy_93821869.pdf_enc9546740.pdf': 'ru',
                    'Optimzed - book_w3vlpeg_235482945.pdf': 'ru',
                    'Optimzed - book_664181550_138791945.pdf': 'en',
                    'Optimzed - 5_999694706.pdf': 'ru',
                    'Optimzed - 104_860776228.pdf': 'ru',
                    'Not Optimized - filipov_etnichekie_protsessy_v_rossijskom_megapolise_i_juzhno__kazahstanskom_regio_PAApkoj.pdf': 'ru',
                    'Not Optimized - book_762283876.pdf_enc4442172.pdf': 'kz',
                    'Not Optimized - book_717820121_346006649.pdf': 'en',
                    'Not Optimized - book_246240674.pdf_enc3941923.pdf': 'kz'}
    lang_to_sub = {'en': '[^A-Za-z]+',
                   'ru': '[^А-Яа-я]+'}
    speaker = {'ru': 'aidar_v2',
               'en': 'lj_16khz'}
    sample_rate = 16_000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = 1
    end = 10
    skip_pages = 2
    max_len = 140


def get_funcs(language):
    speaker = Config.speaker.get(language)
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


def parse_text(text, language):
    parsed_text = re.sub(Config.lang_to_sub.get(language), ' ', text).strip()[:Config.max_len]
    return parsed_text


def parse(args):
    final_json = {}
    files = filter(lambda x: 'ipynb' not in x, os.listdir(args.filepath))

    for filename in files:
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
            if page_num >= Config.skip_pages:
                
                parsing_results = []
                texts = []
                
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        coords = '_'.join(list(map(lambda x: str(round(x)), element.bbox)))
                        response_filename = f'page_{page_num}_coords_{coords}.wav'
                        text = parse_text(element.get_text(), language)
                        if len(text) > 3:
                            texts.append(text)
                            response_json = {
                                "bbox": element.bbox,
                                "text": text,
                                "filename": response_filename
                            }
                            
                            parsing_results.append(response_json)
                audios = get_audios(model, language, texts)
                
                for audio, json_ in zip(audios, parsing_results):
                    save_path = os.path.join(folder_name, json_['filename'])
                    torchaudio.save(save_path, audio.unsqueeze(0), Config.sample_rate)
                    
                book_json[page_num] = parsing_results
        final_json[tail] = book_json
    json_filename = os.path.join(folder_name, f'{folder_name}.json')
    
    with open(json_filename, 'w') as f:
        json.dump(final_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default='PDF_Examples/', help="path to some pdf files")
    args = parser.parse_args()
    parse(args)

