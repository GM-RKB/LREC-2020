# GM-RKB WikiText Error Correction Task (LREC-2020)
GM-RKBWikiText Error Correction Task goal is to benchmark systems that attempt to automatically detect and fix simple typographical errors in WikiText (Wiki Pages).

This repository contains three different datasets used for evaluation, two different models used to fix Wiki pages, along with different tools and test scripts.   
## Example: 
- Original WikiText: <B>Subject Headings:</B> [[Text Corpus]], [[Language Model]]  
- WikiText with Noise: <B>Subject Headings:<B/> [[Text Corpus]], [[Language Model]] 

WikiText with noise is the input of the WikiFixer models. WikiFixer aims at converting the text with noise to its original (clean) form. 

## Usage 

### WikiFixer MLE 
```python
from WikiFixerMLE import WikiFixer
text_noise = '[[Text Corpus]], [[Lnaguage Model]]'
fixer = WikiFixer()
fixer.load_models()
fixer.fix_text(text_noise)
```
```bash
'[[Text Corpus]], [[Language Model]]'
```
### WikiFixer NNet

```python
from WikiFixerNNet import WikiFixer
text_noise = '<B>Subject Headings:<B/> [[Text Corpus]], [[Language Model]]'
fixer = WikiFixer()
fixer.load_models()
fixer.fix_text(text_noise)
```

```bash
'<B>Subject Headings:</B> [[Text Corpus]], [[Language Model]]'
```


## Run tests

### Models Evaluation


### Unit tests

## Directory Structure

## Datasets

### GM-RKB Datasets

### Wikipedia Datasets


```bash
├── WikiFixerMLE.py : WikiFixer MLE Model
├── WikiFixerNNet.py : WikiFixer NNet seq2seq Model
├── model_config.py :  WikiFixer NNet configuration file
├── requirements.txt
├── Datasets
│   ├── MWDump.20191001.Noisetest.parquet :  GM-RKB Wiki dataset
│   ├── Wikipedia.Noise.parquet : Wikipedia Training dataset
│   └── WikipediaTest.Noisetest.parquet : Wikipedia Test dataset
├── mle
│   ├── LanguageModel.py : character based language model implementation
│   ├── model6-0.json
│   ├── model7-1.json
├── nnet
│   ├── data
│   │   └── allowed_chars_sm.json
│   ├── data_processing.py
│   ├── data_vectorization.py
│   └── models
│       ├── gm_rkb_nnet_fixer_GMRKB&Wiki7_sm_e22.h5
│       ├── gm_rkb_nnet_fixer_GMRKB2019_sm_e22.h5
│       ├── gm_rkb_nnet_fixer_GMRKB_PREWiki_sm_e12.h5
│       └── gm_rkb_nnet_fixer_Wikipedia_sm_e10.h5
├── tests
│   ├── log_file.txt
│   ├── multiproc_evaluation.py
│   ├── path.py
│   ├── test_config.py
│   ├── test_integration.py
│   ├── test_mle.py
│   ├── test_model.py
│   ├── test_nnet.py
│   ├── test_script.py
│   └── test_tools.py
└── tools
    ├── WikiTextTools.py
    ├── clean_lm.py
    ├── diff_match_patch.py
    ├── enums.py
    ├── fixer_evaluation.py
    ├── path.py
    ├── review_logs_output.py
    └── test_WikiTextTools.py


```
