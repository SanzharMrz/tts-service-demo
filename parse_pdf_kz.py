import os
import re
import json
import argparse
import warnings

import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model, read_hdf5
from scipy.io.wavfile import write

from nltk import sent_tokenize


class Config:
    max_len = 140
    fs = 22050
    

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filepath", default='/opt/demo_files', help="Path to files")
    args = parser.parse_args()
    return args


def parse_text(text):
    parsed_text = re.sub('([^А-Яа-яa-zA-ZӘәҒғҚқҢңӨөҰұҮүІі-]|[^ ]*[*][^ ]*)', ' ', text).strip()[:Config.max_len]
    return parsed_text


def parse(args):
    files = map(lambda f: os.path.join(args.filepath, f), filter(lambda x: 'ipynb' not in x, os.listdir(args.filepath)))
    head, tail = os.path.split(args.filepath)
    folder_name = os.path.join(head, f'{tail}_parsed')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    folder_json = dict()
    for filename in files:
        print(filename)
        _, tail = os.path.split(filename)
        file_prefix, _ = tail.split('.')

        sub_files = []

        with open(filename, 'r+') as file:
            text = file.readlines()

        # cut it to sentences
        sentences =  sent_tokenize(' '.join(text).replace('\n', ''), language="russian")
        # clear cutted senteces
        clear_sentences = list(map(lambda t: parse_text(t), sentences))
        
        for idx, sent in enumerate(clear_sentences):
            response_filename = f'{file_prefix}_sentid_{idx}.wav'

            with torch.no_grad():
                inference = text2speech(sent.lower())
                wav = vocoder.inference(inference['feat_gen'])

            save_path = os.path.join(folder_name, response_filename)
            write(save_path, Config.fs, wav.view(-1).detach().cpu().numpy())
            sub_files.append(response_filename)
                    
        folder_json[tail] = sub_files
    json_filename = os.path.join(folder_name, f'audio.json')
    with open(json_filename, 'w') as f:
        json.dump(folder_json, f)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    ## specify the path to vocoder's checkpoint
    vocoder_checkpoint="/opt/espnet/egs2/Kazakh_TTS/tts1/exp/vocoder/checkpoint-400000steps.pkl"
    vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
    vocoder.remove_weight_norm()

    ## specify path to the main model(transformer/tacotron2/fastspeech) and its config file
    config_file = "/opt/espnet/egs2/Kazakh_TTS/tts1/exp/tts_train_raw_char/config.yaml"
    model_path  = "/opt/espnet/egs2/Kazakh_TTS/tts1/exp/tts_train_raw_char/train.loss.ave_5best.pth"

    text2speech = Text2Speech(
                        config_file,
                        model_path,
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
    parse(args)
