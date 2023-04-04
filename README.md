text cleaner from https://github.com/CjangCjengh/vits

original repo: https://github.com/jaywalnut310/vits

## Online training and inference
### colab
See [vits-finetuning](https://colab.research.google.com/drive/13FF2pBWxj9rMR1SjI_JpVD6mTRN-kq--?usp=share_link)

# How to use
(Suggestion) Python == 3.7

Only Japanese datasets can be used for fine-tuning in this repo.
## Clone this repository
```sh
git clone https://github.com/SayaSS/vits-finetuning.git
```
## Install requirements
```sh
pip install -r requirements.txt
```
## Download pre-trained model
- [G_0.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0.pth)
- [D_0.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0.pth)
- Edit "model_dir"(line 152) in utils.py
- Put pre-trained models in the "model_dir"/checkpoints

### If you need to customize "n_speakers", please replace the pre-trained model with these two.
- [G_0-p.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/G_0-p.pth)
- [D_0-p.pth](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai/resolve/main/model/D_0-p.pth)

## Create datasets
- Speaker ID should be between 0-803.
- About 50 audio-text pairs will suffice and 100-600 epochs could have quite good performance, but more data may be better. 
- Resample all audio to 22050Hz, 16-bit, mono wav files.
- Audio files should be >=1s and <=10s.
- Audio files scale should be in linear
```
path/to/XXX.wav|speaker id|transcript
```
- Example

```
dataset/001.wav|10|こんにちは。
```
For complete examples, please see filelists/miyu_train.txt and filelists/miyu_val.txt.

## Preprocess
```sh
python preprocess.py --filelists path/to/filelist_train.txt path/to/filelist_val.txt
```
Edit "training_files" and "validation_files" in configs/config.json
## Build monotonic alignment search
```sh
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```
## Train
```sh
# Mutiple speakers
python train_ms.py -c configs/config.json -m checkpoints
```
