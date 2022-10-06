class Config_kz:
    max_len = 140
    fs = 22050
    config_file = "/opt/espnet/egs2/Kazakh_TTS/tts1/exp/tts_train_raw_char/config.yaml"
    model_path = "/opt/espnet/egs2/Kazakh_TTS/tts1/exp/tts_train_raw_char/train.loss.ave_5best.pth"
    vocoder_checkpoint = "/opt/espnet/egs2/Kazakh_TTS/tts1/exp/vocoder/checkpoint-400000steps.pkl"


class Config_ru_en:
    lang_to_sub = {
        'en': '[^A-Za-zА-Яа-я?!.]+',
        'ru': '[^А-Яа-яA-Za-z?!.]+'
    }
    speaker = {
        'ru': 'aidar_v2',
        'en': 'lj_16khz'
    }
    sample_rate = 16_000
    max_len = 130
