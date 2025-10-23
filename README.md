# AnDPro: Accurate KV Cache Eviction via Anchor Direction Projection for Efficient LLM Inference

This repository contains the code for our paper **"Accurate KV Cache Eviction via Anchor Direction Projection for Efficient LLM Inference"**, which has been accepted by NeurIPS 2025.

Paper: [https://openreview.net/pdf?id=Tdl89SZItB](https://openreview.net/pdf?id=Tdl89SZItB)


## Usage of this Repo

### Requirements
The requirements are the same as [AdaKV](https://github.com/FFY0/AdaKV).

```
transformers==4.44.2
flash-attn==2.4.0

datasets
tiktoken
jieba
rouge_score
```

### Installation

```
git clone git@github.com:MIRALab-USTC/LLM-AnDPro.git
cd LLM-AnDPro
make i
```

### Quick Start

To run the experiment on LongBench, you can run the following command:

```bash
cd experiments/LongBench
bash run.sh
```

## Acknowledgement

We extend our gratitude to [AdaKV](https://github.com/FFY0/AdaKV) for their contributions of open-source code. This repository is built upon their latest code framework, with only minor modifications to the token score calculation.

## Citation
If you find this project helpful to your research, please consider citing our paper using the following BibTeX:
```
@inproceedings{
    geng2025accurate,
    title={Accurate {KV} Cache Eviction via Anchor Direction Projection for Efficient {LLM} Inference},
    author={Geng, Zijie and Wang, Jie and Liu, Ziqi and Ju, Feng and Li, Yiming and Li, Xing and Yuan, Mingxuan and Hao, Jianye and Lian, Defu and Chen, Enhong and Wu, Feng},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
}
```