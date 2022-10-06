import argparse
import json
import os
import re
import torch
import warnings
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model, read_hdf5
from scipy.io.wavfile import write

from configs import Config_kz as Config
from util import (parse_text_kz, custom_sent_tokenize)


def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folderpath", default='/opt/demo_files',
                        help="Path to files, please avoid to use '/' at the end")
    args = parser.parse_args()
    return args


def parse(args):
    files = map(lambda f: os.path.join(args.folderpath, f),
                filter(lambda x: 'ipynb' not in x, os.listdir(args.folderpath)))
    head, tail = os.path.split(args.folderpath)
    folder_name = os.path.join(head, f'{tail}_parsed')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_json = dict()
    for filename in files:
        print('PARSING: ', filename)
        _, tail = os.path.split(filename)
        file_prefix, _ = tail.split('.')

        sub_files = []

        with open(filename, 'r+') as file:
            text = file.read()
        # clear text
        text = parse_text_kz(text)

        # cut it to sentences
        clear_sentences = custom_sent_tokenize(text, language="russian", max_len=Config.max_len)

        for idx, sent in enumerate(clear_sentences):
            response_filename = f'{file_prefix}_sentid_{idx}.wav'

            if not sent.strip():
                sent = "Бос өріс"

            with torch.no_grad():
                inference = text2speech(sent.lower())
                wav = vocoder.inference(inference['feat_gen'])

            save_path = os.path.join(folder_name, response_filename)
            write(save_path, Config.fs, wav.view(-1).detach().cpu().numpy())
            sub_files.append({response_filename: sent})

        folder_json[tail] = sub_files
    json_filename = os.path.join(folder_name, f'audio.json')
    with open(json_filename, 'w') as f:
        json.dump(folder_json, f)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    ## specify the path to vocoder's checkpoint
    vocoder = load_model(Config.vocoder_checkpoint).to("cpu").eval()
    vocoder.remove_weight_norm()

    text2speech = Text2Speech(
        Config.config_file,
        Config.model_path,
        device="cpu",
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None
    args = get_args()
    print('START PARSING ...')
    parse(args)
    print('REACHED END OF FOLDER WHILE PARSING ...')
