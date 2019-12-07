# GM-RKB WikiText Error Correction Task (LREC-2020)
GM-RKBWikiText Error Correction Task goal is to benchmark systems that attempt to automatically detect and fix simple typographical errors in WikiText (Wiki Pages).

This repository contains three different datasets used for evaluation, two different models used to fix Wiki pages, along with different tools and test scripts.   
## Example: 
- Original WikiText: ``<B>Subject Headings:</B> [[Text Corpus]], [[Language Model]]``
- WikiText with Noise: ``<B>Subject Headings:<B/> [[Text Corpus]], [[Language Model]]`` 

WikiText with noise is the input of the WikiFixer models. WikiFixer aims at converting the text with noise to its original (clean) form. 

## WikiFixer Usage 

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
The repository contains code for evaluating any system used for Wiki Pages errors fixing. 

- Frist the model has to be defined in `test_config.py`. A class with `fix_text` function using this system has to defined. The following code is an example to add a normal spelling correction tool as a system to be evaluated on the Wiki data [JamSpell](https://github.com/bakwc/JamSpell)

```Python
class jspell(object):
        def __init__(self):
                self.corrector = jamspell.TSpellCorrector()
                self.corrector.LoadLangModel('/mnt/efs/data/en.bin')

        def fix_text(self, text):
                out = []
                for line in text.splitlines():
                        out.append(self.corrector.FixFragment(line))
                return "\n".join(out)
```              
- Second, using the same file, the configuration of the test can be set as following 
```python 
    def test_models(self):
        log_file = './output/log_file.csv' # output file

        models = ["mle"] # model to be evaluated
        datafiles = ["./Datasets/MWDump.20191001.Noisetest.parquet"] #dataset used for evaluation 
        for datafile in datafiles:
            for model in models:
                 config = test_config.get_config(datafile=datafile,Model=model) #load configuraiton
                config["sample_size"] = 100 #number of pages used in the evaluation porcess
  
                Emetric, score, types_stats = test_script.run_test(config)
                l = [datafile, model, Emetric[0], Emetric[1], Emetric[2],metric[3], score]
                with open(log_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(l)
```
- third to run the test using command `python -m unittest tests.test_model` 
### Unit tests
 A number of unit testing scripts are available:
 - `test_tools` unit testing for WikiTools, which is a class containing different functions used in the process of running and evaluating different models.
 - `test_mle` unit testing for WikiFixer MLE
 - `test_nnet` unit testing for WikiFixer NNet

They can be run using the command `python -m unittest tests.test_tools` (this is example to run the first unit testing script).  
 

## Datasets

### GM-RKB Dataset

### Wikipedia Dataset

## Directory Structure

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
