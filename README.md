# Official Implementation of AHAMask
> This is the official repository of "AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions"
> Accepted to AAAI 2026

Since difference LALMs are implemented differently, we take [SALMONN](https://arxiv.org/abs/2310.13289) as an example here.

[![arXiv](https://img.shields.io/badge/arXiv-2509.01787-b31b1b.svg)](https://arxiv.org/abs/2509.01787)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🛠️ Environment

This repo does not imply additional requirements to the official SALMONN repository. Please use `pip install -r requirements.txt` for your environment.

⚡Note: This repo also supports Ascend NPU devices.

## 🎯 Inference with pre-trained masks

We release the AHAMasks trained for different acoustic tasks on SALMONN in `./legacy_checkpoints/`, as they are so small 😛!
The task and datasets can be inferred from the directory naming in most cases. For the composite tasks:
* `asr_gender_librispeech` and `gender_asr_librispeech` means "ASR|GR" and "GR|ASR" tasks, respectively.
* `asr_gender_json_librispeech` means Json output format with "ASR" and "GR" keys.

First, download all necessary checkpoints following [here](https://github.com/bytedance/SALMONN/tree/salmonn?tab=readme-ov-file#-how-to-inference-in-cli). Then, batched inference with a trained mask checkpoint can be done by:
```bash
python evaluate.py -c configs/decode_config.yaml
# It should generate ./tmp.json that contains gender recognition results on the provided samples.
```
Please change the `test_data_path`, `output_path`, and `weight_mask_path` for your case. The `test_data_path` should be a path to a json file, whose schema should be the same as `data/test/demo.json`. Note that `prompt` in the config yaml should be set to empty if you specify a `weight_mask_path`; otherwise, it falls back to the original SALMONN model, hence `prompt` should be specified to a non-empty instruction.

In inference, AHAMasks are supposed to be binary. But the released checkpoints are continuous mask logits. In most cases, we threshold them by 0 before model forwarding.

Each checkpoint is essentially a 1-D tensor with shape `(n_layers x n_heads + n_layers)`. For each layer, there are a mask for each of `n_heads` heads and the additional FFN mask. In this paper, the FFN mask is not used. Hence, the 2-D representation of a mask can be obtained by `mask.reshape(n_layers, n_heads+1)[:, :-1]`.

For interactive inference, you can also use
```bash
python cli_inference.py -c configs/decode_config.yaml
```

## 🏋️ Train
To train an AHAMask, first prepare a training data json. It should have the same schema as `data/test/demo.json`. 
Please then set the `train_ann_path` in `configs/config.yaml` to the path of this json file.

Training can be achieved by
```bash
# Single GPU
python train.py -c configs/config.yaml
# or in distributed environment
python -m torch.distributed.run --nproc_per_node=<ngpu> --master_port=<port> train.py -c configs/config.yaml
```

## 📚 Citation & Acknowledgement
This project contains substantial portions of code from the following two sources:
* Most of this repo is the same as [bytedance/SALMONN](https://github.com/bytedance/SALMONN)
* For the attention head masking: [OpenDFM/HeadsUp](https://github.com/OpenDFM/HeadsUp)

All credits for the overall codebase go to the original authors.

Please consider citing:
```bibtex
@inproceedings{guo2026ahamask,
  title={AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions},
  author={Guo, Yiwei and Li, Bohan and Wang, Hankun and Li, Zhihan and Wang, Shuai and Chen, Xie and Yu, Kai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```