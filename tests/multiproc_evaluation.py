import os
import sys
import csv
sys.path.insert(0, '../tools/')
sys.path.insert(0, '../')

from multiprocessing import Queue, Process

from tools.WikiTextTools import WikiTextTools
from model_config import CONFIG, LOGGER
from WikiFixerNNet import WikiFixerNNet

from fixer_evaluation import get_diff_log, get_diff_distance, get_eval_metrics, compare_diff_logs
from test_script import compare_diff_logs_timeout

text_test_noise = []


def get_test_dataset(datafile=None,k=[],i1=None,i2=None,sample_size=None):
    global text_test_noise
    global text_test_org
    tools = WikiTextTools()
    if datafile:
        df = tools.read_parquet_df(datafile)
    else:
        df = tools.read_parquet_df(CONFIG.DATASET_FILE_NAME)
    if sample_size:
        df =  df.sample(sample_size)
        text_test_noise = df["pageTextNoisy"].tolist()[i1: i2]
        text_test_org= df["contentsOriginal"].tolist()[i1: i2]
    else:
        if k:
            df_valid = df[df['k'] in k]
        else:
            df_valid = df[df['k'] == CONFIG.k]

        text_test_noise = df_valid["pageTextNoisy"].tolist()[i1:i2]
        text_test_org = df_valid["contentsOriginal"].tolist()[i1: i2]


class FixerNNet_Worker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue
        self.fixer = WikiFixerNNet()

    def run(self):
        import csv
        # set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)

        # load models
        self.fixer.load_model()

        LOGGER.info('s2snet init done %s' % self._gpuid)
        csv_file = open(CONFIG.PREDICTION_CSV_FILE_NAME, mode='a')
        fieldnames = ['i', 'Emetric','score']  # 'p']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        while True:
            rowX = self._queue.get()
            if rowX is None:
                self._queue.put(None)
                break
            LOGGER.info('Woker {} Starting index {}'.format(self._gpuid, str(rowX['i'])))
            p_ind = rowX['i']
            p_text = text_test_noise[p_ind]
            p_otext= text_test_org[p_ind]          

            fixed_text = self.fixer.fix_text(p_text)

            original_logs = get_diff_log(p_text, p_otext)
            fixed_logs = get_diff_log(p_text, fixed_text)
            ot = compare_diff_logs_timeout(original_logs, fixed_logs)

            TP, FN, FP, DN = len(ot[0]), len(ot[1]), len(ot[2]), len(ot[3])
            score = TP-5.0*FP

            #writer.writerow({'i': rowX['i'], 'p': fixed_text})
            writer.writerow({'i': rowX['i'], 'Emetric': [TP, FN, FP, DN],'score':score})
            LOGGER.info('Woker {} Ending index {}'.format(self._gpuid, str(rowX['i'])))

        LOGGER.info('s2snet done %s' % self._gpuid)


class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(FixerNNet_Worker(gpuid, self._queue))

    def start(self):

        # put all of files into queue
        for i in range(len(text_test_noise)):
            self._queue.put({"i": i})

        # add a None into queue to indicate the end of task
        self._queue.put(None)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        LOGGER.info("all of workers have been done")


gpus_list = ['GPU:0',
             'GPU:1',
             'GPU:2',
             'GPU:3',
             'GPU:4',
             'GPU:5',
             'GPU:6',
             'GPU:7']


def init_pred_file():
    csv_file = open(CONFIG.PREDICTION_CSV_FILE_NAME, mode='w')
    fieldnames = ['i', 'p']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.close()


def run(gpuids):
    # create the preds csv file
    init_pred_file()
    # get test set
    get_test_dataset(sample_size=10)
    # init scheduler
    scheduler = Scheduler(gpuids)
    # start processing and wait for complete
    scheduler.start()


if __name__ == '__main__':
    run(gpus_list)
