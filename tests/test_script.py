import sys
sys.path.insert(0, 'tools/')
from fixer_evaluation import get_diff_log, get_diff_distance, get_eval_metrics, compare_diff_logs
from functools import wraps
import os
from path import get_path
import errno
import signal
import operator
from WikiTextTools import WikiTextTools
from WikiFixer import WikiFixer
from tqdm import tqdm



def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


@timeout(600)
def compare_diff_logs_timeout(original_logs, fixed_logs):
    return compare_diff_logs(original_logs, fixed_logs)


tools = WikiTextTools()
"""data"""


def run_test(config={"fixer": None, "sample_size": 1000, "i1": None, "i2": None, "k": None, "data_file": "/mnt/efs/data/MWDump.20181118.Noise.parquet", }):
    
    i1,i2 = config["i1"], config['i2']
    fixer = config["fixer"]
    #data_file = "/home/ubuntu/WikiFixer/tools/enwiki3.20190601.Noisetest.parquet"
    df = tools.read_parquet_df(config['data_file'])
    if config['sample_size']:
        df = df.sample(config['sample_size'])

    if config['k']:
        df = df[df['k'] == config['k']]

    text_test_org = df["contentsOriginal"].tolist()[i1:i2]
    text_test_noise = df["pageTextNoisy"].tolist()[i1:i2]

    id_test = df["id"].tolist()[i1:i2]
    size_test = df["size"].tolist()[i1:i2]

    """Wikifixer model
    """

    Emetric = [0, 0, 0, 0, 0]
    score = 0
    s = {"type": [0, 0, 0, 0, 0], "chars": [], "pos": []}
    types_stats = [0, 0, 0, 0, 0]
    info = [s]*4
    logs = [[], []]
    for testi in tqdm(range(len(text_test_org))):
        ntext = str(text_test_noise[testi])
        otext = str(text_test_org[testi])
        size = size_test[testi]
        Id = id_test[testi]
        out = []
        Ls = []
        out = fixer.fix_text(ntext)
        
        original_logs = get_diff_log(ntext, otext)
        fixed_logs = get_diff_log(ntext, out)
        original_logs2 = get_diff_log(otext, ntext)
        for log in original_logs2:
            types_stats[log["type"]]+=1
        try:
            ot = compare_diff_logs_timeout(original_logs, fixed_logs)
        except:
            ot = [[], [], [], []]
        '''
        for oti in range(len(ot)):
            info[oti]["chars"] = list(set(info[oti]["chars"]))
            for elem in ot[oti]:
                info[oti]["pos"].append([elem["pos"], size])
                info[oti]["type"][elem["type"]] += 1
                types_stats[elem["type"]] += 1
                info[oti]["chars"].extend(elem["chars"])
        '''
        TP, FN, FP, DN = len(ot[0]), len(ot[1]), len(ot[2]), len(ot[3])
        FB_logs = ot[2]
        TP_logs = ot[0]

        Emetric[0] += TP
        Emetric[1] += FN
        Emetric[2] += FP
        Emetric[3] += DN
        Emetric[4] += size-sum([TP, FN, FP, DN])
        score += TP-5.0*FP

    return Emetric, score,types_stats
