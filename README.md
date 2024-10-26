<p align="center">

  <h2 align="center">[Interspeech'24] GTR-Voice: Articulatory Phonetics Informed Controllable Expressive Speech Synthesis </h2>
  <p align="center">
    <strong>Zehua Kcriss Li</strong></a><sup>1</sup>
    · 
    <strong>Meiying Melissa Chen</strong></a><sup>1</sup>
    · 
    <strong>Yi Zhong</strong></a><sup>1</sup>
    ·
    <strong>Pinxin Liu</strong></a><sup>2</sup>
    ·
    <strong>Zhiyao Duan</strong></a><sup>1</sup>
    ·
    <br>
    <sup>1</sup>Department of Electrical and Computer Engineering, University of Rochester  &nbsp;&nbsp;&nbsp; <sup>2</sup>Department of Computer Science, University of Rochester
    <br>
    <br>
    </br>

  </p>
    </p>
<div align="center">
  <img src="./assets/teaser.jpg" alt="Realistic Gesture"></a>
</div>

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

