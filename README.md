<p align="center">

  <h2 align="center">[Interspeech'24] GTR-Voice: Articulatory Phonetics Informed Controllable Expressive Speech Synthesis </h2>
  <p align="center">
    <strong>Pinxin Liu</strong></a><sup>1</sup>
    · 
    <strong>Pengfei Zhang</strong></a><sup>2</sup>
    · 
    <strong>Hyeongwoo Kim</strong></a><sup>4</sup>
    ·
    <strong>Pablo Garrido</strong></a><sup>3</sup>
    ·
    <strong>Ari Shapiro</strong></a><sup>3</sup>
    ·
    <br><strong>Kyle Olszewski</strong></a><sup>3</sup>
    ·  
    <br>
    <sup>1</sup>University of Rochester  &nbsp;&nbsp;&nbsp; <sup>2</sup>University of California, Irvine
    <br>
    <sup>3</sup>Flawless AI   &nbsp;&nbsp;&nbsp; <sup>4</sup>Imperial College London
    <br>
    </br>

  </p>
    </p>
<div align="center">
  <img src="./assets/teaser.jpg" alt="Realistic Gesture"></a>
</div>

## 📣 News
* **[2024.10.07]** Release training and inference code with instructions to preprocess the [PATS](https://chahuja.com/pats/download.html) dataset.

* **[2024.10.12]** Release paper.

## 🗒 TODOs
- [x] Release the Dataset.
- [x] Build the Github Page.
- [x] Release pretrained weights.
- [x] Release training code.
- [x] Release code about evaluation metrics.
- [ ] Release the presentation video.

## ⚒️ Environment
We recommend a python version ```>=3.9``` and cuda version ```=11.8```. It's possible to have other compatible version.

```bash
conda create -n gtr python=3.10
conda activate gtr
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

We test our code on NVIDIA A10, NVIDIA A100, NVIDIA A6000.

## ⭕ Quick Start
I have already provided the pretrained weights in the ```inference/ckpt``` folder. 
```image-gen5.tar.pth``` is the pretrained weights for the image-warping and image-refinement. 
```train-300.pt``` is the pretrained weights for the mask gesture generator. ```train-500.pt``` is the pretrained weights for residual gesture generator.

Download [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and put it into the ```inference/data/wavlm``` folder.

Now, get started with the following code:

```bash
cd inference
CUDA_VISIBLE_DEVICES=0 python inference.py --wav_file ./assets/001.wav --init_frame ./assets/001.png
```

## 🔥 Train Your Own Model


### Train StyleTTS
Change into the ```styletts``` folder:

```bash
cd ../styletts
```

Then slice and preprocess the data:

```bash
cd data 
python 
cd ..
```

Run the following code to train StyleTTS stage 1

```bash

```

Run the following code to train StyleTTS stage 2
  
```bash

```


### Train the Image-Warping Module
Change into the ```image-warping``` folder:

```bash
cd image-warping
```

Then run the following code to train the image-warping:

```bash 
accelerate launch run.py --config config/img-warp.yaml --mode train
```


### Training the Edge-heatmap based Refinement Network
Change into the ```image-refine``` folder:

```bash
cd ../image-refine
```


## 🙏 Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [StyleTTS](https://github.com/yl4579/StyleTTS)
* [FastPitch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)

