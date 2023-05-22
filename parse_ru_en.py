import argparse
import json
import os
import re
import torch
import torchaudio

from configs import Config_ru_en as Config
from util import (parse_text, chunks_f, custom_sent_tokenize)


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
    for i in range(len(texts)):
        if not texts[i].strip():
            if language == 'en':
                texts[i] = "Empty field"
            else:
                texts[i] = "Пустое поле"
    if language == 'en':
        audios = Config.apply_tts(texts=texts,
                                  model=model,
                                  sample_rate=Config.sample_rate,
                                  symbols=Config.symbols,
                                  device=Config.device)
    elif language == 'ru':
        audios = [model.apply_tts(texts=[text], sample_rate=Config.sample_rate)[0] for text in texts]
    return audios


def parse(args):
    files = map(lambda f: os.path.join(args.folderpath, f),
                filter(lambda x: 'ipynb' not in x, os.listdir(args.folderpath)))
    language = args.lang
    head, tail = os.path.split(args.folderpath)
    folder_name = os.path.join(head, f'{tail}_parsed')

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_json = dict()

    model = get_funcs(language)
    tok_lang = 'russian' if language == 'ru' else 'english'

    for filename in files:
        print('PARSING: ', filename)
        _, tail = os.path.split(filename)
        file_prefix, _ = tail.split('.')

        with open(filename, 'r+') as file:
            text = file.read()
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # parse it
        text = parse_text(text, language)

        # cut it to sentences
        clear_sentences = custom_sent_tokenize(text, language=tok_lang, max_len=Config.max_len)
        # Only leave sentences with at least 2 alphabetic characters
        clear_sentences = [sent for sent in clear_sentences if len(re.findall(r'[a-zA-Zа-яА-Я]', sent)) > 1]

        # create response
        sub_files = []
        audios = []
        for chunk_ in chunks_f(clear_sentences, 5):
            audios.extend(get_audios(model, language, chunk_))

        for sent_id, (sent, audio) in enumerate(zip(clear_sentences, audios)):
            response_filename = f'{file_prefix}_sentid_{sent_id}.wav'
            save_path = os.path.join(folder_name, response_filename)
            torchaudio.save(save_path, audio.unsqueeze(0), Config.sample_rate)
            sub_files.append({response_filename: sent})

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.device = device
    args = get_args()
    print('START PARSING ...')
    parse(args)
    print('REACHED END OF FOLDER WHILE PARSING ...')
