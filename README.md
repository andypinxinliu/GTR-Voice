<p align="center">

  <h2 align="center">[Arxiv'24] Realistic-Gesture: Co-Speech Gesture Video Generation Through Context-aware Gesture Representation </h2>
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
- [x] Release data preprocessing code.
- [x] Release inference code.
- [x] Release pretrained weights.
- [x] Release training code.
- [x] Release code about evaluation metrics.
- [ ] Release the presentation video.

## ⚒️ Environment
We recommend a python version ```>=3.9``` and cuda version ```=11.8```. It's possible to have other compatible version.

```bash
conda create -n s2g python=3.10
conda activate s2g
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

## 📊 Data Preparation
Due to copyright considerations, we are unable to directly provide the preprocessed data subset mentioned in our paper. Instead, we provide the filtered interval ids and preparation instructions. 

To get started, please download the meta file ```cmu_intervals_df.csv``` provided by [PATS](https://chahuja.com/pats/download.html) (you can fint it in any zip file) and put it in the ```data-preparation``` folder. Then run the following code to prepare the data.

```bash
cd data-preparation
bash prepare_data.sh
```
After running the above code, you will get the following folder structure containing the preprocessed data:

```bash
|--- data-preparation
|    |--- data
|    |    |--- img
|    |    |    |--- train
|    |    |    |    |--- chemistry#99999.mp4
|    |    |    |    |--- oliver#88888.mp4
|    |    |    |--- test
|    |    |    |    |--- jon#77777.mp4
|    |    |    |    |--- seth#66666.mp4
|    |    |--- audio
|    |    |    |--- chemistry#99999.wav
|    |    |    |--- oliver#88888.wav
|    |    |    |--- jon#77777.wav
|    |    |    |--- seth#66666.wav
```

## 🔥 Train Your Own Model
Here we use [accelerate](https://github.com/huggingface/accelerate) for distributed training.



### Train the Latent Gesture Generator
Change into the ```gesture_pattern_generation``` folder:

```bash
cd ../gesture_pattern_generation
```

Download [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and put it into the ```data/wavlm``` folder.
Then slice and preprocess the data:

```bash
cd data 
python create_dataset_gesture.py --stride 0.4 --length 3.2 --keypoint_folder ../../stage1/feature --wav_folder ../../data-preparation/data/audio --extract-baseline --extract-wavlm
cd ..
```
Run the following code to train gesture-clip

```bash
accelerate launch train_clip.py --batch_size 32 --max_epoch 100 --lr 0.0001 --gamma 0.05 --feature_name face

accelerate launch train_clip.py --batch_size 32 --max_epoch 100 --lr 0.0001 --gamma 0.05 --feature_name body
```

Run the following code to train the gesture vector quantization
  
```bash
accelerate launch train_vq.py --num_quantizers 4  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --feature_name face

accelerate launch train_vq.py --num_quantizers 4  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.05 --feature_name body
```

Run the following code to train the gesture mask generator:

```bash
accelerate launch train_mask.py --cross
```

Run the following code to train the residual gesture generator:

```bash
accelerate launch train_residual.py --cross
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

Download ```mobile_sam.pt``` provided by [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and put it in the ```pretrained_weights``` folder. Then extract bounding boxes of hands for weighted loss (only training set needed):
  
```bash
python get_bbox.py --img_dir ../data-preparation/data/img/train
```

Now you can train the refinement network:

```bash
accelerate launch run.py --config config/refine.yaml --mode train --tps_checkpoint ../stage1/log/stage1.pth.tar
```


## 🙏 Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [Thin-Plate Spline Motion Model for Image Animation](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)
* [S2G-Diffusion](https://github.com/thuhcsi/S2G-MDDiffusion)

