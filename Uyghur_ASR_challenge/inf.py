#nohup python inf.py > /DATA/ai20resch11003/uyghur-asr/train_log2.txt 2>&1 &
#pkill -f inf.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from peft import PeftModel

DEVICE = "cuda:0"
BASE_MODEL = "ixxan/whisper-small-uyghur-thugy20"
CKPT_DIR = "/DATA/ai20resch11003/uyghur-asr/whisper-uyghur-peft/checkpoint-570"

feature_extractor = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
tokenizer = WhisperTokenizer.from_pretrained(BASE_MODEL)
processor = WhisperProcessor.from_pretrained(BASE_MODEL)

base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
base_model.config.forced_decoder_ids = None
base_model.config.suppress_tokens = []

model = PeftModel.from_pretrained(base_model, CKPT_DIR)
model = model.merge_and_unload().to(DEVICE)
model.eval()

DATA_DIR = "/DATA/ai20resch11003/uyghur-asr"
WAV_DIR = os.path.join(DATA_DIR, "wavs")
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_df["filepath"] = test_df["filepath"].apply(lambda x: os.path.join(WAV_DIR, os.path.basename(x)))

print("Generating transcriptions for Test Dataset...")
predictions, ids = [], []
batch_size = 64

def load_and_extract(fp):
    audio, _ = librosa.load(fp, sr=16000)
    return feature_extractor(audio, sampling_rate=16000).input_features[0]

for i in range(0, len(test_df), batch_size):
    batch_df = test_df.iloc[i : i + batch_size]

    # parallelize disk I/O + feature extraction across CPU threads instead of serial loop
    with ThreadPoolExecutor(max_workers=8) as ex:
        input_features_list = list(ex.map(load_and_extract, batch_df["filepath"]))

    input_tensor = torch.tensor(np.array(input_features_list)).to(DEVICE)  # <-- now actually on GPU

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_ids = model.generate(
            input_features=input_tensor,
            max_new_tokens=225,
            num_beams=5,
        )

    transcriptions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend(transcriptions)
    ids.extend(batch_df["ID"].tolist())
    print(f"{i + len(batch_df)}/{len(test_df)} done")

submission_df = pd.DataFrame({"ID": ids, "transcription": predictions})
submission_df["transcription"] = submission_df["transcription"].str.strip()
submission_df.to_csv("/DATA/ai20resch11003/uyghur-asr/submission.csv", index=False)
print("File generated successfully..")