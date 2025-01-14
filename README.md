# TabLaP: Numerical Problem Solver for Tabular Question Answering
This project is focused on improving the performance of LLMs for numerical problems and its reliability over tabular data.
![Model Overview](model_overview.pdf)


## Create Virtual Environment
Create a conda environment named TabLaP 
```
conda env create -f environment.yaml
```

## Python Path
If you find  errors regarding pkg path, please run following command before the scripts
```
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
```

## Reproducing Experiment Results
To reproduce the experiment results, please follow these steps:
### WTQ dataset
Navigate to the wtq directory and run the evaluation scripts
For evaluate NumSolver performance:
```bash
python wtq/evaluate.py
```

For evaluate TabLaP performance:
```bash
python wtq/tablap_eval.py
```

### FTQ dataset
Navigate to the ftq directory and run the evaluation script:
For evaluate NumSolver performance:
```bash
python ftq/evaluate.py
```

For evaluate TabLaP performance:
```bash
python wtq/tablap_eval.py
```

### TabFact_small dataset
Navigate to the tabfact directory and the detailed codes are in the "TabFact_small.ipynb" jupyternotebook 
Notes:
The codes for fine-tuning Tapex-large and OmniTab-large using your own datasets are also included in the notebook. Due to the file size limit, the train.jsonl file is uploaded in my shared link (same as the link with model ckpt below).

## Fine-tuning or Inferring by yourself
1. First you need to build up your own training or testing dataset. Your can refer to the scripts named "building_testing_data.py" and "building_training_data.py" in wtq or ftq directories.

2. After obtaining your dataset, for fine-tuing (LoRA), you can try:
```bash
python ftq/model_ft.py
python wtq/model_ft.py
```
For inference, you can try:
```bash
python ftq/inference.py
python wtq/inference.py
```
Besides, we also include our checkpoints for AnsSelector and TwEvaluator in the "example_ckpt" directory (please download using this link: https://drive.google.com/drive/folders/1Ss2ia1NswGZw1xEsHexouS4IU43ojAy6?usp=drive_link).
We share the same AnsSelector and TwEvaluator for both datasets, since the tasks are simialr, however, you can further fine-tuning the modules for specific datasets.

## Contact
If you have any further questions, please email yuxiang.wang8@student.unimelb.edu.au

