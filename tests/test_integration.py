import unittest
import time
import tests.test_config as test_config
import tests.test_script as test_script
import sys
import getopt
import argparse


class TestWikiFixerInteg(unittest.TestCase):

    log_file = 'log_file.txt'
    f = open(log_file, "w")

    base_wikitext = {'Emetric': [
        2454, 10961, 90, 56, 2820609], 'score': 2010.0}
    base_wikitpedia = {'Emetric': [1130, 14832,
                                   677, 46, 3380815], 'score': - 2247.0}

    def test_wikitextdata(self):
        Emetric, score = test_script.run_test(test_config.get_config())
        print(Emetric, score)
        self.assertTrue(score >= self.base_wikitext['score'])
        self.assertTrue(Emetric[0] >= self.base_wikitext['Emetric'][0])
        self.assertTrue(Emetric[2] <= self.base_wikitext['Emetric'][2])

    def test_wikitpedia(self):       
        config= test_config.get_config()
        config['data_file'] = '/mnt/efs/data/MWDumpWikipediaFull.20191001.Noisetest.parquet'
        config['i1']=500
        config['i2']=750
        Emetric, score = test_script.run_test(config)
        print(Emetric,score)
        self.assertTrue(score >= self.base_wikitpedia['score'])
        self.assertTrue(Emetric[0] >= self.base_wikitpedia['Emetric'][0])
        self.assertTrue(Emetric[2] <= self.base_wikitpedia['Emetric'][2])
  
