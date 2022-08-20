# tts-service-demo

# Usage

```bash
# For Kazakh lang
cd /opt/tts-service-demo
python parse_kz.py --help
python parse_kz.py --folderpath /opt/demo_files # sorry but avoid "/" as last symbol, promise to fix it
```

Small guide of other stt-tts services installation

# NU SAIDA_KAZAKH_ASR [KAZAKH TTS]

install espnet as preequisite
```bash
# install cmake
sudo apt-get install cmake

# install sox
sudo apt-get install sox

# install sndfile
sudo apt-get install libsndfile1-dev

# install ffmpeg
sudo apt-get install ffmpeg

# install flac
sudo apt-get install flac

# clone espnet repo
cd <any-place>
git clone https://github.com/espnet/espnet

# setup venv
cd <espnet-root>/tools
./setup_venv.sh $(command -v python3)

# install espnet
cd <espnet-root>/tools
make

# optional with cuda
cd <espnet-root>/tools
make TH_VERSION=1.10.1 CUDA_VERSION=11.3

# If you don’t have nvcc command, packages are installed for **CPU** mode by default. If you’ll turn it on manually, give CPU_ONLY option.
cd <espnet-root>/tools
make CPU_ONLY=0

# go to asr dir
cd ISSAI_SAIDA_Kazakh_ASR/asr1
ln -s ../../../tools/kaldi/egs/wsj/s5/steps steps
ln -s ../../../tools/kaldi/egs/wsj/s5/utils utils

# create exp
./setup_experiment.sh <exp-name>

# download weights
wget https://issai.nu.edu.kz/wp-content/uploads/2020/10/model.tar.gz

# untar
tar --zxvf model.tar.gz -C ksc/<exp-name>/

# To decode a single audio, specify paths to the following files inside recog_wav.sh script:

lang_model=  rnnlm.model.best
cmvn=  data/train/cmvn.ark # path to cmvn.ark
recog_model= model.last10.avg.best # path to e2e model, in case of transformer

# Then, run the following script:
./recog_wav.sh <path-to-audio-file>
```

# VOSK [KAZAKH AND RUSSIAN STT]

Installtation process

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git
sudo apt-get install bc
sudo apt-get install g++
sudo apt-get install zlib1g-dev make automake autoconf bzip2 libtool subversion
sudo apt-get install libatlas3-base

# Install Kaldi:
git clone https://github.com/kaldi-asr/kaldi.git kaldi –origin upstream
cd kaldi/tools
extras/check_dependencies.sh
make

# To install the irst language model:
cd kaldi/src
./configure –shared
make depend
make

# Inorder to install the language model kaldi_lm:
cd kaldi/tools
extras/install_kaldi_lm.sh

# Install VOSK
git clone https://github.com/alphacep/vosk-api
cd vosk-api/src
KALDI_ROOT=<KALDI_ROOT> make
cd ../vosk-api/python
python3 setup.py install

# Install models and run recognition
cd vosk-api/python/example
wget https://alphacephei.com/vosk/models/vosk-model-kz-0.15.zip
unzip vosk-model-kz-0.15.zip
mv vosk-model-kz-0.15 model
python3 ./test_simple.py test.wav # your wav file
```

# HuggingSound and transformers [RUSSIAN STT]

Super simple usage

## HuggingSound

install one lib

```bash
pip install huggingsound
```

Start recognition
```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

## Transformers

install few libs
```bash
pip install librosa
pip install datasets
pip install transformers
```
Start your recognition script

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "ru"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
SAMPLES = 5

test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)
```

