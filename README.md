# GM-RKB WikiText Error Correction Task (LREC-2020)

## Directory Structure
```bash
├── WikiFixer.py : WikiFixer MLE Model
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
