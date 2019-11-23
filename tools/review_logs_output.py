import pandas as pd
import numpy as np
import json
from WikiTextTools import WikiTextTools
from fixer_evaluation import get_diff_log, compare_diff_logs

tools = WikiTextTools()

df = tools.read_parquet_df("/mnt/efs/data/MWDump.20181118.Noisez2.parquet")

for i in range(df.shape[0]):
    orig_text = df.iloc[i]['contentsOriginal']
    noise_text = df.iloc[i]['pageTextNoisy']
    log = df.iloc[i]['log']

    #just to make new lines appear in text (like how it was added to the parquet file)
    open("o.txt", 'w').write(orig_text)
    open("n.txt", 'w').write(noise_text)
    orig_text = "\n".join(open('o.txt', encoding='utf-8').readlines())
    noise_text = "\n".join(open('n.txt', encoding='utf-8').readlines())
    
    # get the diff_log using the new method 
    errors_pos = get_diff_log(orig_text, noise_text)

    try:
        TP, FN, FP, DN = compare_diff_logs(json.loads(log.replace("\'", "\"")), errors_pos)
        if FN or FP:
            print(i)
            print('=' * 200)
            print('=' * 200)
            print('=' * 200)
            print(log)
            print('=' * 200)
            print(errors_pos)
        open('delta_log.txt', 'a').write('=' * 200)
        open('delta_log.txt', 'a').write('=' * 200)
        open('delta_log.txt', 'a').write('=' * 200)
        open('delta_log.txt', 'a').write(str(i))

        try:
            open('delta_log.txt', 'a').write(log)
        except:
            pass
    except:
        pass
