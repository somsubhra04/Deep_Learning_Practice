## Overview

The NPPE problem statement invites participants to build a Transformer based text classification system capable of automatically categorizing legal and policy documents into their appropriate categories. The competition focuses on applying modern Natural Language Processing techniques to long, information rich documents where careful preprocessing and model adaptation play an important role.

The objective of this competition is to develop a strong understanding of Transformer fine tuning using encoder based language models. Participants are expected to explore how different tokenization strategies, preprocessing techniques, and parameter efficient fine tuning methods influence model performance on a challenging real world document classification task.

Unlike short text classification problems, the documents in this challenge vary considerably in length and writing style. Since most Transformer models have a limited context window, participants must carefully design their preprocessing pipeline before fine tuning. Decisions such as document truncation, text cleaning, learning rate selection, and hyperparameter optimization can significantly affect the final performance.

The competition emphasizes practical machine learning skills rather than simply training a pretrained model. Participants are encouraged to experiment with different encoder architectures, compare fine tuning strategies, and systematically analyze the impact of their design choices.

This challenge provides an opportunity to apply concepts learned throughout the course in a realistic setting while working under practical compute constraints similar to those encountered in industry.

Participants are encouraged to:

- Build an efficient preprocessing and tokenization pipeline.
- Fine tune pretrained Transformer models for sequence classification.
- Explore parameter efficient fine tuning methods such as LoRA and QLoRA.
- Optimize hyperparameters including learning rate, batch size, and training schedule.
- Compare different encoder architectures and analyze their performance.
- Develop models that generalize well to previously unseen documents.
- Success in this competition depends not only on selecting a strong pretrained model but also on making informed decisions throughout the entire machine learning pipeline, from data preparation to model optimization and evaluation.

### Evaluation Metric
Submissions are evaluated using the Accuracy Score

Submission Format
Participants must submit a CSV file with the following columns:

- ID: Unique identifier for each sample
- label: Predicted label

### Dataset Description
The dataset consists of legal and policy related documents collected from multiple sources and organized into several document categories.

Each document contains rich textual content with varying length and writing style. Some documents are concise, while others contain detailed legal language and technical terminology.

The provided files are:

- train.csv

Contains labeled training samples.

Columns:

id, Unique document identifier
text, Document content
label, Target category

- test.csv

Contains unlabeled documents for which predictions must be generated.

Columns:

id
text
