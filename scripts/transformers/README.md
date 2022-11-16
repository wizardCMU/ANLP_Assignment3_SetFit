# Transformers Baselines (Updated by Weizhe 11/14/2022)

This folder contains the scripts used to train the 🤗 Transformers baselines quoted in the SetFit paper.

## Setup

To run the scripts, first create a Python virtual environment, e.g. with `conda`:

```
conda create -n baselines-transformers python=3.9 && conda activate baselines-transformers
```

Next, install the required dependencies (Modified by Weizhe)

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers[sentencepiece,optuna]==4.20.0
pip install -U scikit-learn
pip install typer
pip install setfit
```

## Usage

### Fewshot finetuning (Modified by Weizhe)

To finetune a pretrained model on all the test datasets used in SetFit, run:

Notice that we may only want to get the results for |N|=8 and |N|=64. In this case, please modify the 15th line in miniconda3/envs/baslines-transformers/lib/python3.9/site-packages/setfit/data.py: The SAMPLE_SIZES can be changed to SAMPLE_SIZES = [8, 64]

```
python run_fewshot.py train-all-datasets --model-id=roberta-large --batch-size=4
```

### Full finetuning (Modified by Weizhe)

To finetune a pretrained model on all the test datasets used in SetFit, run:

```
python run_full.py train-all-datasets --model-id=roberta-large --batch-size=24
```

Or we can run it one by one:

```
python run_full.py train-single-dataset --dataset-id=emotion --metric=accuracy --model-id=roberta-large --learning-rate=2e-5 --batch-size=24
rm -rf checkpoints
python run_full.py train-single-dataset --dataset-id=sst5 --metric=accuracy --model-id=roberta-large --learning-rate=2e-5 --batch-size=24
rm -rf checkpoints
python run_full.py train-single-dataset --dataset-id=SentEval-CR --metric=accuracy --model-id=roberta-large --learning-rate=2e-5 --batch-size=24
rm -rf checkpoints
python run_full.py train-single-dataset --dataset-id=amazon_counterfactual_en --metric=matthews_correlation --model-id=roberta-large --learning-rate=2e-5 --batch-size=24
rm -rf checkpoints
python run_full.py train-single-dataset --dataset-id=ag_news --metric=accuracy --model-id=roberta-large --learning-rate=2e-5 --batch-size=4
rm -rf checkpoints
python run_full.py train-single-dataset --dataset-id=enron_spam --metric=accuracy --model-id=roberta-large --learning-rate=2e-5 --batch-size=8
rm -rf checkpoints
```

### Multilingual finetuning

We provide three different ways to run SetFit in multilingual settings:

* `each`: train on data in target language
* `en`: train on English data only
* `all`: train on data in all languages

To finetune a baseline in one of these setting, run:

```
python run_fewshot_multilingual.py train-single-dataset \
--model-id=xlm-roberta-base \
--dataset-id=amazon_reviews_multi_en \
--metric=mae \
--learning-rate=2e-5 \
--batch-size=4 \
--multilinguality=each
```

To finetune a baseline on all the multilingual test sets in the paper, run:

```
python run_fewshot_multilingual.py train-all-datasets \
    --model-id=xlm-roberta-base \
    --learning-rate=2e-5 \
    --batch-size=4 \
    --multilinguality=each
```
