# CTIR (Learning Class-Transductive Intent Representations for Zero-shot Intent Detection)
Here is the implementation of our IJCAI-2021 [Learning Class-Transductive Intent Representations for Zero-shot Intent Detection](https://arxiv.org/pdf/2012.01721.pdf).
This repository contains code modified from [here for CapsNet+CTIR](https://github.com/nhhoang96/ZeroShotCapsule-PyTorch-), [here for ZSDNN+CTIR](https://github.com/xuandif-cmu/Zero-shot-DNN) and [here for +LOF+CTIR](https://github.com/thuiar/DeepUnkID), many thanks!

# Train & Test (CapsNet+CTIR)
```bash
cd capsnet-CTIR
```
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
