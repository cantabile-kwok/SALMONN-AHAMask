# -----------------------------------------------------------------------------
# Copyright (c) 2025 Yiwei Guo
# Licensed under the MIT License
# -----------------------------------------------------------------------------

import argparse
# In Ascend NPU environment, we need the following import. Otherwise, will enter 'except' branch.
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass
import torch
from transformers import WhisperFeatureExtractor
import os
from config import Config
from models.salmonn import SALMONN
from utils import get_dataloader, prepare_sample
from tqdm import tqdm
from dataset import SALMONNDataset
import json


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", '-c', type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

prob_threshold = cfg.config.get("prob_threshold", 0.5)

if hasattr(cfg.config.model, "weight_mask_path") and (cfg.config.model.weight_mask_path is not None):
    weight_mask = torch.load(cfg.config.model.weight_mask_path, map_location="cpu")['model']['tensor']

    weight_mask.to(args.device)
    print("AHAMask weight is", weight_mask)
    print("After thresholding, ", 
          (torch.sum(weight_mask.sigmoid()>=prob_threshold)/weight_mask.numel()).item() * 100,
          "% of attention heads will be activated.")
else:
    print("Not using mask: official model")
    weight_mask = None

test_data_path = cfg.config.test_data_path
output_file = cfg.config.output_path
dir_path = os.path.dirname(output_file)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

test_dataset = SALMONNDataset(test_data_path, cfg.config.model.whisper_path)
test_dataloader = get_dataloader(test_dataset, is_train=False, config=cfg.config, use_distributed=False)

test_result = dict()
test_result['prompt'] = cfg.config.prompt
test_result['prediction'] = list()

for batch_samples in tqdm(test_dataloader):
    batch_samples = prepare_sample(batch_samples)
    prompt = cfg.config.prompt
    
    prompt = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
    ] * len(batch_samples['spectrogram'])
    answer = model.generate(batch_samples, cfg.config.generate, prompts=prompt, weight_tensor=weight_mask, prob_threshold=prob_threshold)

    for i in range(len(batch_samples['spectrogram'])):
        answer_i = answer[i].strip().replace("\n", " ")
        if answer_i.endswith("</s>"):
            answer_i = answer_i[:-4]

        test_result['prediction'].append(
            {
                "id": batch_samples['id'][i],
                "ground_truth": batch_samples['text'][i],
                "output": answer_i
            }
        )

with open(output_file, 'w') as fw:
    json.dump(test_result, fw, indent=4, ensure_ascii=False)

print(f"Successfully written file to {output_file}")
