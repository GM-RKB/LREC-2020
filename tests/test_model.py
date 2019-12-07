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
        log_file = '/home/ubuntu/WikiFixer/tests/output/log_file.csv'

        models = ["nnet"]
        datafiles = ["/mnt/efs/data/MWDump.20191001.Noisetest.parquet"]
        for datafile in datafiles:
            for model in models:
                config = test_config.get_config(model, datafile)
                Emetric, score, types_stats = test_script.run_test(config)
                print(types_stats)
                l = [datafile, model, Emetric[0], Emetric[1], Emetric[2],
                     Emetric[3], score, config["i1"], config["i2"], config["k"], types_stats]
                with open(log_file, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(l)



#"/mnt/efs/data/wiki/WikipediaTest.Noisetest.parquet",
#"/mnt/efs/data/MWDump.20191001.Noisetest.parquet"
#/mnt/efs/data/MWDumpWikipediaPart.20191001.Noisetest.parquet
