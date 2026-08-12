#nohup python train.py > /DATA/ai20resch11003/uyghur-asr/train_log.txt 2>&1 &
#pkill -f train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
import numpy as np
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, LoraModel
import jiwer

from huggingface_hub import login
HF_TOKEN = "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"
login(token=HF_TOKEN)

MODEL_NAME = "ixxan/whisper-small-uyghur-thugy20"

DATA_DIR = "/DATA/ai20resch11003/uyghur-asr"
WAV_DIR  = os.path.join(DATA_DIR, "wavs")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
test_csv_path  = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_csv_path)
test_df  = pd.read_csv(test_csv_path)

# Map filenames to full path
train_df['filepath'] = train_df['filepath'].apply(lambda x: os.path.join(WAV_DIR, os.path.basename(x)))
test_df['filepath']  = test_df['filepath'].apply(lambda x: os.path.join(WAV_DIR, os.path.basename(x)))

print(f"Loaded Train Samples: {len(train_df)}")
print(f"Loaded Test Samples:  {len(test_df)}")

# Filter extreme outliers (max ~63s vs 95th pctile ~19.5s)
import librosa as lb
train_df["duration"] = train_df["filepath"].apply(lambda f: lb.get_duration(path=f))
train_df = train_df[train_df["duration"] <= 20.0].reset_index(drop=True)
train_df = train_df.drop(columns=["duration"])

print(f"Train: {len(train_df)}")

# Initialize Processor & Tokenizer
# Omit language and task for unsupported languages like Uyghur ('ug')
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

# preprocessing function
def prepare_dataset(batch):
    audio, _ = librosa.load(batch["filepath"], sr=16000)
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]
    
    if "transcription" in batch and pd.notna(batch["transcription"]):
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

print("Extracting Mel-Spectrogram features...")
train_dataset = train_dataset.map(
    prepare_dataset, 
    remove_columns=train_dataset.column_names
)
test_dataset = test_dataset.map(
    prepare_dataset, 
    remove_columns=[c for c in test_dataset.column_names if c != "ID"] 
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if "labels" in features[0]:
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = jiwer.cer(label_str, pred_str)
    return {"cer": cer}

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

# SpecAugment — helps generalization on small data
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_feature_prob = 0.05


peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],  # wider than just q/v
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.gradient_checkpointing_enable()
model.config.use_cache = False

model.print_trainable_parameters()

#Training model
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uyghur-peft",
    per_device_train_batch_size= 32,
    gradient_accumulation_steps= 1,# Effective Batch Size = 32
    learning_rate= 1e-3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs= 5,
    gradient_checkpointing= False,
    fp16=True,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="none"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting Fine-Tuning...")
trainer.train()

#Inference on test data
print("Generating transcriptions for Test Dataset...")
model.eval()

predictions = []
ids = []
batch_size = 16

for i in range(0, len(test_df), batch_size):
    batch_df = test_df.iloc[i : i + batch_size]
    
    input_features_list = []
    for fp in batch_df['filepath']:
        audio, _ = librosa.load(fp, sr=16000)
        feat = feature_extractor(audio, sampling_rate=16000).input_features[0]
        input_features_list.append(feat)
        
    input_tensor = torch.tensor(np.array(input_features_list))
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_tensor,
            max_new_tokens=225,
            num_beams=5
        )
        
    transcriptions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    predictions.extend(transcriptions)
    ids.extend(batch_df['ID'].tolist())

# Format output file
submission_df = pd.DataFrame({
    'ID': ids,
    'transcription': predictions
})

# Basic text cleanup
submission_df['transcription'] = submission_df['transcription'].str.strip()

submission_path = "/DATA/ai20resch11003/uyghur-asr/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("File generated successfully..")