# -*- coding: utf-8 -*-
from WikiTextTools import WikiTextTools
import pandas as pd
import os
files=[]
files2=[]
#files2 = ['/mnt/efs/data/rkb-mediawiki-20191001-1210.xml']
i=0
path = '/mnt/efs/data/wiki/wikitest'
for filename in os.listdir(path):
    if not filename.endswith('.xml'):
        continue
    fullname = os.path.join(path, filename)
    files.append(fullname)

files2.extend(files)
print(files2)
input(">")
#data_file = "/mnt/efs/data/MWDump.20181118.Noisez0.parquet"
wikitextool = WikiTextTools()  # NoiseVariables={
#"noise_level": 0.05, "swap_rate": 0.25, "delete_rate": 1.00, "add_rate": 0.01, "change_rate": 0.01})
wikitextool.add_noise_to_xml(
    files2, "/mnt/efs/data/wiki/WikipediaTest.Noisetest.parquet", k=10)
#wikitextool.split_File("t1.txt")
'''
df = wikitextool.read_parquet_df(data_file)
lol = df["contentsOriginal"].tolist()
l=0
l+=1
for each in lol:
    if "Ã" in each:
        l+=1

print(l)
'''

#print(len(df))
#print(wikitextool.evaluate_log([{"pos": 5, "type": 3, "chars": ["j"]}, {"pos": 12, "type": 4, "chars": ["j", "k"]}], [
#    {"t": "4", "p1": 11, "p2": 11, "history": "wwe", "char": "j", "char2": "j", "k": "k",
#                         "L": "wwe", "R": "wwc"}]))
'''
text = open("t1.txt").read()
print(len(wikitextool.add_noise_to_text(text)[1]))
text = open("t2.txt").read()
print(len(wikitextool.add_noise_to_text(text)[1]))
'''

'''wikipedia full 31
'''

