import unittest
import time
import tests.test_config as test_config
import tests.test_script as test_script
import sys
import getopt
import argparse
import csv


class TestWikiFixerInteg(unittest.TestCase):


    def test_models(self):
        log_file = './output/log_file.csv'

        models = ["mle"]
        datafiles = ["./Datasets/MWDump.20191001.Noisetest.parquet"]
        for datafile in datafiles:
            for model in models:
                config = test_config.get_config(datafile=datafile,Model=model)
                config["sample_size"] = 100
  
                Emetric, score, types_stats = test_script.run_test(config)
                l = [datafile, model, Emetric[0], Emetric[1], Emetric[2],Emetric[3], score]
                with open(log_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(l)


