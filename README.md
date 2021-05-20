# CTIR (Learning Class-Transductive Intent Representations for Zero-shot Intent Detection)
Here is the implementation of our IJCAI-2021 [Learning Class-Transductive Intent Representations for Zero-shot Intent Detection](https://arxiv.org/pdf/2012.01721.pdf).  

The Appendix mentioned in this paper are shown in [Appendix.pdf](https://github.com/PhoebusSi/CTIR/blob/master/Appendix.pdf)

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
The main idea of two-stage method for GZSID is to ﬁrst determine whether an utterance belongs to unseen intents (i.e., *Y_seen* ), and then classify it into a speciﬁc intent class. This method bypasses the need to classify an input sentence among all the seen and unseen intents, thereby alleviating the domain shift problem. To verify the performance of integrating CTIR into the two-stage method, we design a new two-stage pipeline (`+LOF+CTIR` ). In Phase 1, a test utterance is classiﬁed into one of the classes from *Y_seen ∪ { y_unseen }* using the density-based algorithm LOF(LMCL) (refer [here](https://github.com/thuiar/DeepUnkID)). In Phase2, we perform ZSID for the utterances that have been classiﬁed into *y_unseen* , using the CTIR methods such as `CapsNet+CTIR`, `ZSDNN+CTIR`.
