# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# Original Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------------
#
# Modifications Copyright (c) 2025 Yiwei Guo
# Licensed under the MIT License
# -----------------------------------------------------------------------------


import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
# In Ascend NPU environment, we need the following import. Otherwise, will enter 'except' branch.
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cfg-path", type=str, required=True, help='path to configuration file')
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

prob_threshold=cfg.config.get("prob_threshold", 0.5)

if hasattr(cfg.config.model, "weight_mask_path"):
    weight_mask = torch.load(cfg.config.model.weight_mask_path, map_location="cpu")['model']['tensor']
    print(f"Using weight mask from {cfg.config.model.weight_mask_path}.")

    weight_mask.to(args.device)
    print("After thresholding, ", 
          (torch.sum(weight_mask.sigmoid()>=prob_threshold)/weight_mask.numel()).item() * 100,
          "% of attention heads will be activated.")
else:
    weight_mask = None
    print(f"Not using weight masks, i.e. in default official SALMONN mode.")

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

while True:
    try:
        print("=====================================")
        wav_path = input("Your Wav Path:\n")
        if weight_mask is None:
            prompt = input("Your Prompt:\n")
        else:
            prompt = ""

        samples = prepare_one_sample(wav_path, wav_processor)
        prompt = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]
        print("Output:")
        print(model.generate(samples, cfg.config.generate, prompts=prompt, weight_tensor=weight_mask, prob_threshold=prob_threshold)[0])
    except Exception as e:
        print(e)
