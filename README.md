# CTIR (Learning Class-Transductive Intent Representations for Zero-shot Intent Detection)
Here is the implementation of our IJCAI-2021 [Learning Class-Transductive Intent Representations for Zero-shot Intent Detection](https://arxiv.org/pdf/2012.01721.pdf).
![image](https://github.com/PhoebusSi/CTIR/blob/main/model.jpg)

This repository contains code modified from [here for CapsNet+CTIR](https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-), [here for ZSDNN+CTIR](https://github.com/xuandif-cmu/Zero-shot-DNN) and [here for +LOF+CTIR](https://github.com/thuiar/DeepUnkID), many thanks!
# Download the Glove embedding file
```bash
cd data/nlu_data
```
You can download the Glove embedding file we used from [here](https://drive.google.com/file/d/1Qqy_HnymjakddEUgWxoYQm74VKLWeOON/view?usp=sharing). 
# Train & Test (CapsNet+CTIR)
```bash
cd capsnet-CTIR
```
![image](https://github.com/PhoebusSi/CTIR/blob/main/hyperparameter4capsnet%2BCTIR.jpg)
## Train & Test for SNIP dataset in the ZSID setting
```bash
python main.py SNIP ZSID
```
## Train & Test for SNIP dataset in the GZSID setting
```bash
python main.py SNIP GZSID
```
## Train & Test for CLINC dataset in the ZSID setting
```bash
python main.py CLINC ZSID
```
## Train & Test for CLINC dataset in the GZSID setting
```bash
python main.py CLINC GZSID
```
# Train & Test (ZSDNN+CTIR)
```bash
cd zerodnn-CTIR
```
![image](https://github.com/PhoebusSi/CTIR/blob/main/hyperparameter4zeroshotdnn%2BCTIR.jpg)
## Train & Test for SNIP dataset in the ZSID setting
```bash
python zerodnn_main.py SNIP ZSID
```
## Train & Test for SNIP dataset in the GZSID setting
```bash
python zerodnn_main.py SNIP GZSID
```
## Train & Test for CLINC dataset in the ZSID setting
```bash
python zerodnn_main.py CLINC ZSID
```
## Train & Test for CLINC dataset in the GZSID setting
```bash
python zerodnn_main.py CLINC GZSID
```
# Train & Test (+LOF+CTIR in the GZSID setting)
