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

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from nltk import sent_tokenize


NUM_CHUNKS = 6


class Config:
    file_to_lang = {'SCANNED - 4108_18411976.pdf': 'kz',
                    'SCANNED - 4110_353652508.pdf': 'kz',
                    'Optimzed - tanirbergenova_aaberikbaeva_ma_inzhenerlіk_mehanika_-_2__334400008_pdf_enc6407430': 'kz',
                    'Optimzed - sotsialnoe_gosudarstvo_teorija_metodologija_mehanizmy_93821869.pdf_enc9546740.pdf': 'ru',
                    'Optimzed - book_w3vlpeg_235482945.pdf': 'ru',
                    'Optimzed - book_664181550_138791945.pdf': 'en',
                    'Optimzed - 5_999694706.pdf': 'ru',
                    'Optimzed - 104_860776228.pdf': 'ru',
                    'Not Optimized - filipov_etnichekie_protsessy_v_rossijskom_megapolise_i_juzhno__kazahstanskom_regio_PAApkoj.pdf': 'ru',
                    'Not Optimized - book_762283876.pdf_enc4442172.pdf': 'kz',
                    'Not Optimized - book_717820121_346006649.pdf': 'en',
                    'Not Optimized - book_246240674.pdf_enc3941923.pdf': 'kz'}
    start = 1
    end = 1000
    skip_pages = 0
    max_len = 140
    fs = 22050
    

def get_args():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filepath", default='PDF_Examples/', help="path to some pdf files")
    args = parser.parse_args()
    return args


def parse_text(text):
    parsed_text = re.sub('([^А-Яа-яa-zA-ZӘәҒғҚқҢңӨөҰұҮүІі-]|[^ ]*[*][^ ]*)', ' ', text).strip()[:Config.max_len]
    return parsed_text


def parse(args):
    files = filter(lambda x: 'ipynb' not in x, os.listdir(args.filepath))

    for filename in files:
        filename = 'PDF_Examples/Optimzed - tanirbergenova_aaberikbaeva_ma_inzhenerlіk_mehanika_-_2__334400008_pdf_enc6407430.pdf'
        head, tail = os.path.split(filename)
        folder_name, _ = tail.split('.')
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
                    "ybox": ((NUM_CHUNKS - i - 1) / NUM_CHUNKS * page_height, (NUM_CHUNKS - i) / NUM_CHUNKS * page_height),
                }
                for i in range(NUM_CHUNKS)
            ]

            for current_chunk_num, chunk in enumerate(chunks):
                #collect raw text
                raw_page_text = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        for line in element:
                            chunk_num = int(((page_height - (line.bbox[1] + line.bbox[3]) / 2) / page_height) * NUM_CHUNKS)
                            if chunk_num != current_chunk_num:
                                continue
                            raw_text = line.get_text()
                            clear_text = parse_text(raw_text)
                            if len(clear_text) > 10:
                                raw_page_text.append(raw_text)

                # cut it to sentences
                sentences = list(filter(lambda sent: len(sent) > 10, sent_tokenize(' '.join(raw_page_text).replace('\n', ''), language="russian")))
                # clear cutted senteces
                clear_sentences = list(map(lambda t: parse_text(t), sentences))
                for idx, sent in enumerate(clear_sentences):
                    response_filename = f'page_{page_num}_sentid_{idx}.wav'
                    with torch.no_grad():
                        _, c_mel, *_ = text2speech(sent.lower())
                    wav = vocoder.inference(c_mel)
                    save_path = os.path.join(folder_name, response_filename)
                    write(save_path, Config.fs, wav.view(-1).detach().cpu().numpy())
                    chunk['audios'].append(response_filename)
            book_json[page_num] = {
                "height": page_height,
                "chunks": chunks
            }
        json_filename = os.path.join(folder_name, f'audio.json')
        with open(json_filename, 'w') as f:
            json.dump(book_json, f)


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    ## specify the path to vocoder's checkpoint
    vocoder_checkpoint="/home/s.murzakhmetov/audio/tts/espnet/exp/vocoder/checkpoint-400000steps.pkl"
    vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
    vocoder.remove_weight_norm()

    ## specify path to the main model(transformer/tacotron2/fastspeech) and its config file
    config_file = "/home/s.murzakhmetov/audio/tts/espnet/exp/tts_train_raw_char/config.yaml"
    model_path  = "/home/s.murzakhmetov/audio/tts/espnet/exp/tts_train_raw_char/train.loss.ave_5best.pth"

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
