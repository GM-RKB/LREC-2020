# -*- coding: utf-8 -*-

import xml.etree.ElementTree as etree
import zipfile
import tarfile
import gzip
import shutil
import os

import time

import sys
import string
import random
import numpy.random
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from itertools import zip_longest as zip

from Levenshtein import distance
from fixer_evaluation import get_diff_log

import logging
logger = logging.getLogger('wikitextlog')

# $%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c
# "#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\r'
#CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

symbols = ' !#$%&()*+,-./:;<=>?@[]{}'
digits = '0123456789'
UpperAlpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LowerAlpha = 'abcdefghijklmnopqrstuvwxyz'
CHARS = symbols + digits + UpperAlpha + LowerAlpha


class WikiTextTools(object):

    def __init__(self, NoiseVariables=None):
        if NoiseVariables:
            self.NoiseVariables = NoiseVariables
        else:
            self.NoiseVariables = {"noise_level": 0.005, "swap_rate": 0.25,
                                   "delete_rate": 0.25, "add_rate": 0.25, "change_rate": 0.25}
        self.update_rates()

    def update_rates(self):
        """Updates the rates so they are all cover the range between 0 and 1, to be able to use on random number to decide having error and its type 
        all rates should be less than 1 and their sum must be less than 1
        """
        temp = 0.0
        for key in ["swap_rate", "delete_rate", "add_rate", "change_rate"]:
            temp += self.NoiseVariables[key]
            self.NoiseVariables[key] = temp

    def decompress(self, FilePath, ExtractionDir="."):
        """Decompress files of type: zip, tar, tar.gz, tgz, tar.bz2, tbz, gz
        FilePath is the input compressed file
        ExtractionDir is the directory of the uncompressed file
        """
        FilePath = os.path.abspath(FilePath)
        opener, mode = None, None
        if FilePath.endswith("zip"):
            opener, mode = zipfile.ZipFile, "r"
        elif FilePath.endswith(".tar"):
            opener, mode = tarfile.open, "r:"
        elif FilePath.endswith(".tar.gz") or FilePath.endswith('.tgz'):
            opener, mode = tarfile.open, "r:gz"
        elif FilePath.endswith(".tar.bz2") or FilePath.endswith('.tbz'):
            opener, mode = tarfile.open, "r:bz2"
        elif FilePath.endswith(".gz"):
            filename, file_extension = os.path.splitext(FilePath)
            outfile = os.path.join(ExtractionDir, filename)
            try:
                with gzip.open(FilePath, 'rb') as f_in:
                    with open(outfile, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                logger.error('Failed to decompress file: ' + str(e))
        else:
            raise Exception("compressed file '%s' not supported" % FilePath)

        if opener and mode:  # in case file is (tar or zip)
            try:
                tf = opener(FilePath, mode)
                tf.extractall(path=ExtractionDir)
                tf.close()
            except Exception as e:
                logger.error('Failed to decompress file: ' + str(e))

    def append_to_parquet_table(self, dataframe, filepath=None, writer=None):
        """Method writes/append dataframes in parquet format.

        This method is used to write pandas DataFrame as pyarrow Table in parquet format. If the methods is invoked
        with writer, it appends dataframe to the already written pyarrow table.

        :param dataframe: pd.DataFrame to be written in parquet format.
        :param filepath: target file location for parquet file.
        :param writer: ParquetWriter object to write pyarrow tables in parquet format.
        :return: ParquetWriter object. This can be passed in the subsequenct method calls to append DataFrame
            in the pyarrow Table
        """
        table = pa.Table.from_pandas(dataframe)
        if writer is None:
            writer = pq.ParquetWriter(filepath, table.schema)
        writer.write_table(table=table)
        return writer

    def read_parquet_df(self, filename):
        """reads parquet file into pandas dataframe
        """
        table = pq.read_table(filename)
        return table.to_pandas()

    def get_page_data(self, Page):
        """extract data from page (xml)
        text content in the page is in "contentsOriginal"
        returns data structured in a pandas dataframe format
        """
        pageTitle = Page['title']
        pageText = Page['text']
        pageRedirect = Page['redirect']

        concept = 0
        if (re.match('m/Category\:Concept/', pageText)):
            concept = 1

        publication = 0
        if (re.match('m/Category\:Publication/', pageText)):
            publication = 1

        person = 0
        if (re.match('m/Category\:Person/', pageText)):
            person = 1

        redirect = 0
        if (re.match('m/\#REDIRECT/', pageText)):
            redirect = 1

        if (re.match('m/Category:Malformed/', pageText)):
            # return empty dataframe (same use as continue but the loop was removed)
            return pd.DataFrame()

        mydata = [
            {'pageTitle': pageTitle, 'contentsOriginal': pageText, 'contentsRedirect': pageRedirect,
             'concept': concept, 'publication': publication, 'person': person, 'redirect': redirect}, ]
        df = pd.DataFrame(mydata)
        df["concept"] = df['concept'].astype('int')
        df["publication"] = df['publication'].astype('int')
        df["person"] = df['person'].astype('int')
        df["redirect"] = df['redirect'].astype('int')

        return df

    def write_pageXML_parquet(self, inputFilePath, outputFilePath):
        """writes the text in xml file in a parquate file without any change
        """
        writer = None
        i = 0
        for Page in GetAllPages(inputFilePath):

            df = self.get_page_data(Page)
            df.insert(1, 'id', i)
            i += 1
            if not df.empty:
                writer = self.append_to_parquet_table(
                    df, outputFilePath, writer)
        if writer:
            writer.close()

    def add_noise_to_text(self, text):
        """adds noise to text and return text with noise and log of noise added
        """
        flag = True  # this flag is used to ensure not having two consecutive errors(noise)
        #types = {"1": "swap", "2": "insertion", "3": "deletion", "4": "change"}
        character_distribution = len(CHARS)*[1.0/len(CHARS)]
        noisyText = []
        #the following two variables are used to make sure the position of each character is updated with respect of the lines
        I = 0
        ln = 0

        for original_line in text.splitlines(True):
            #this way the length of each line (ln) is saved and added to the total count (I)
            I += ln
            characters = str(original_line)
            if len(characters) == 0:
                ln = 1
            else:
                ln = len(characters)

            i = 0
            noisyLine = []

            while i < len(characters):  # iterate over every character
                # if character not an English character or we reached end of the text
                if characters[i] not in list(CHARS) or i > len(characters)-1:
                    noisyLine.append(characters[i])
                    i += 1
                    continue

                # condition to limit the noise to certain rate using a generated random number
                if random.random() < self.NoiseVariables["noise_level"]:
                    if flag:
                        rand_number = random.random()
                        # swap
                        if (rand_number <= self.NoiseVariables["swap_rate"] and i < len(characters)-1):
                            if characters[i+1] != "\n":
                                # print("s")
                                c1, c2 = characters[i+1], characters[i]
                                noisyLine.append(c1)
                                noisyLine.append(c2)

                                i += 1
                            else:
                                noisyLine.append(characters[i])
                        # delete
                        elif (rand_number <= self.NoiseVariables["delete_rate"]):
                            # print("d")
                            c = characters[i]

                        # add
                        elif (rand_number <= self.NoiseVariables["add_rate"]):
                            # print("a")
                            noisyCharacter = numpy.random.choice(
                                list(CHARS), 1, p=character_distribution)[0]
                            noisyLine += noisyCharacter
                            noisyLine.append(characters[i])

                        # replace
                        else:
                            # print("c")
                            c = characters[i]

                            noisyCharacter = numpy.random.choice(
                                list(CHARS), 1, p=character_distribution)[0]
                            noisyLine.append(noisyCharacter)

                        flag = False
                        i += 1
                        continue

                    else:
                        noisyLine.append(characters[i])
                else:
                    # no noise added
                    noisyLine.append(characters[i])

                i += 1
                flag = True

            if noisyLine:
                if noisyLine[-1] != "\n":
                    # noisyLine.append("\n")
                    pass
            noisyText.append("".join(noisyLine))
        #if noisyText and len(noisyText[-1]) > 1:
        #    noisyText[-1] = noisyText[-1][:-1]
        return ("".join(noisyText))

    def add_noise_to_xml(self, inputFilePath, outputFilePath, k=1, page_limit=False):
        """read the xml file and add noise to the text
        write the output in parquet file
        """
        #chrs=[]
        writer = None
        i = 0
        for fil in inputFilePath:
            for Page in GetAllPages(fil):
                print(i)
                if page_limit:
                    if i == page_limit:
                        break
                df = self.get_page_data(Page)
                #chrs += list(df["contentsOriginal"][0])
                if not df.empty:
                    df.insert(1, 'id', i)
                    i += 1
                    kvalue = i % k
                    contentsOriginal = df["contentsOriginal"][0]
                    pageTextNoisy = self.add_noise_to_text(contentsOriginal)

                    log = 0  # get_diff_log(pageTextNoisy, contentsOriginal)
                    # self.evaluate(contentsOriginal, pageTextNoisy)[0]
                    dis = 0
                    df.insert(1, 'pageTextNoisy', pageTextNoisy)
                    df.insert(1, 'size', len(pageTextNoisy))
                    df.insert(1, 'distance', dis)
                    df.insert(1, 'log', str(log))
                    df.insert(1, 'k', kvalue)

                    writer = self.append_to_parquet_table(
                        df, outputFilePath, writer)
        if writer:
            writer.close()

    def analyze_parquet(self, inputFilePath):
        parquet_file = pq.ParquetFile(inputFilePath)
        return parquet_file.metadata, parquet_file.schema

    def levenshtein(self, s1, s2):
        return distance(s1, s2)

    def evaluate(self, text1, text2):
        """evaluate two documents by comparing each line, word by word in the same order 
        this is an initial version where order matters.
        future work would be more features. For exmaple, to detect if there is a line inserted in a file shifting the whole document by one line.
        """
        Doc1Count = len(list(text1))
        Doc2Count = len(list(text2))
        count = self.levenshtein(text1, text2)

        return count, Doc1Count, Doc2Count

    def merge_logs(self, logs):
        if logs:
            new_log = []
            for log in logs:
                if log:
                    new_log += log
            return sorted(new_log, key=lambda i: i['p1'], reverse=False)
        else:
            return []

        return new_log

    def evaluate_log(self, log_org, log_fixed):
        pos_change = 0
        FIX_pos = 0

        true_pos = 0
        false_pos = 0
        false_neg = 0

        det_nofix = 0

        f_iter = iter(log_fixed)
        o_iter = iter(log_org)

        fnext = True
        onext = True

        FB_logs = []

        def next_org(o_iter):
            org = next(o_iter)

            org_pos = org["pos"]
            org_chars = org["chars"]
            org_type = org["type"]
            return org_pos, org_chars, org_type

        def next_fix(f_iter):
            fix = next(f_iter)
            fix_pos = fix["p1"]
            fix_type = fix["t"]
            fix_history = fix["history"]
            if fix_type == "1":
                fix_chars = [fix["char"], fix["char2"]]
            elif fix_type == "2":
                fix_chars = [fix["char"]]
            elif fix_type == "3":
                fix_chars = [fix["k"]]
            elif fix_type == "4":
                fix_chars = [fix["char"], fix["k"]]
            return fix_pos, fix_chars, fix_type, [fix_history, fix["char"], fix["char2"], fix["k"], fix_type, fix_pos]

        while fnext or onext:

            if fnext:
                try:
                    fix_pos, fix_chars, fix_type, fix_l = next_fix(f_iter)
                    fix_type = int(fix_type)
                    FIX_pos = fix_pos
                except:
                    fix_pos, fix_chars, fix_type = None, None, None
            if onext:
                try:
                    org_pos, org_chars, org_type = next_org(o_iter)
                    ORG_POS = org_pos
                except:
                    org_pos, org_chars, org_type = None, None, None

            fnext, onext = False, False
            if fix_pos == None and org_pos == None:
                fnext, onext = False, False
                continue
            elif fix_pos == None or org_pos == None:
                fnext, onext = True, True
                if fix_pos == None:
                    false_neg += 1
                    #print("false_neg", [fix_pos, fix_chars, fix_type], [
                    #      org_pos, org_chars, org_type], pos_change)
                else:
                    false_pos += 1
                    FB_logs.append(fix_l)
                    #print("false_pos", [fix_pos, fix_chars, fix_type], [
                    #      org_pos, org_chars, org_type], pos_change)
                continue

            fix_pos = FIX_pos+pos_change
            if [fix_pos, fix_chars, fix_type] == [org_pos, org_chars, org_type]:
                true_pos += 1
                fnext, onext = True, True
                #print("true pos", [fix_pos, fix_chars, fix_type], [
                #    org_pos, org_chars, org_type], pos_change)
            else:
                # and not rightToLeft) or (fix_pos < org_pos and rightToLeft ) :
                if (fix_pos > org_pos):
                    false_neg += 1
                    if org_type == 3:
                        pos_change += 1
                    elif org_type == 2:
                        pos_change -= 1
                    onext = True
                    #print("false_neg", [fix_pos, fix_chars, fix_type], [
                    #      org_pos, org_chars, org_type], pos_change)

                # and not rightToLeft) or (fix_pos > org_pos and  rightToLeft):
                elif (fix_pos < org_pos):
                    false_pos += 1
                    FB_logs.append(fix_l)
                    #print("2", [fix_pos, fix_chars, fix_type], [org_pos, org_chars, org_type],pos_change)
                    if fix_type == 3:
                        pos_change -= 1
                    elif fix_type == 2:
                        pos_change += 1
                    #print("false_pos", [fix_pos, fix_chars, fix_type], [
                    #      org_pos, org_chars, org_type], pos_change)
                    fnext = True
                elif fix_pos == org_pos:
                    if fix_type != org_type:
                        det_nofix += 1
                        fnext, onext = True, True
                        if org_type == 3:
                            pos_change += 1
                        elif org_type == 2:
                            pos_change -= 1
                        if fix_type == 3:
                            pos_change -= 1
                        elif fix_type == 2:
                            pos_change += 1
                        #print("none", [fix_pos, fix_chars, fix_type], [
                        #    org_pos, org_chars, org_type], pos_change)
            #print(pos_change)
        #print( [true_pos, false_neg, false_pos, det_nofix])

        return [true_pos, false_neg, false_pos, det_nofix, FB_logs]

        files = []
        for i in range(k):
            files.append(open(re.sub("(\.txt)", str(i)+"\\1",
                                     data_file), mode="w", encoding="utf-8"))
        with open(data_file, mode="r", encoding="utf-8") as dataf:
            line = dataf.readline()
            while line:
                line = dataf.readline()


def strip_tag_name(elem):
    t = elem.tag
    idx = k = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


class GetAllPages(object):
    def __init__(self, inputFilename):
        Parser = etree.XMLParser(
            html=0, target=etree.TreeBuilder(), encoding="utf-8")
        self.tokenParser = etree.iterparse(
            inputFilename, events=('start', 'end'), parser=Parser)
        self.currentPage = {}

    def __iter__(self):
        return self

    def __next__(self):
        for event, elem in self.tokenParser:
            tagName = strip_tag_name(elem)

            if event == 'end':
                if tagName == 'page':
                    # Found the end of </page> block
                    ret = self.currentPage
                    self.currentPage = {}

                    # Make sure that "text" is always populated, even for empty pages (0 bytes of text).
                    if not ret.get('text'):
                        ret['text'] = ""

                    if not ret.get('redirect'):
                        ret['redirect'] = ""

                    return ret
                elif tagName == 'title':
                    self.currentPage['title'] = elem.text
                elif tagName == 'text':
                    self.currentPage['text'] = elem.text
            elif event == 'start':
                if tagName == 'redirect':
                    self.currentPage['redirect'] = elem.attrib['title']

        # Entire XML dump was parsed, nothing more to read
        raise StopIteration

    def next(self):
        return self.__next__()
