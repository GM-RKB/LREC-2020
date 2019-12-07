import sys
import numpy as np
from keras.models import load_model

sys.path.insert(0, './tools')
from tools.fixer_evaluation import get_diff_log
from model_config import CONFIG, LOGGER
from nnet.data_processing import process_text
from nnet.data_vectorization import vectorize, get_char_table
import spacy


class WikiFixerNNet(object):

    def __init__(self):
        self.ctable = get_char_table()

    def load_model(self):
        print(CONFIG.SAVED_MODEL_FILE_NAME)
        self.model = load_model(CONFIG.SAVED_MODEL_FILE_NAME)
        LOGGER.info("NNet Model loaded.")

    def fix_text(self, text):
        """ 
         Using the nnet model, construct the fixed text from the processed sequences 
        :param text: the noise text to be fixed 
        Parameters to be used in this process: 
        :param model: the trained nnet model 
        :param ctable: the CharacterTable object used to encode/decode the text 
        :param clipped_ids: ids in the sequences for unnatural cuts 
        :return: constructed fixed text 
        """
        LOGGER.info("Processing text..")
        text_test_noise_seqs, text_test_clipped_seq_ids = process_text(text)
        LOGGER.info("Text processed. Converted into {} sequences".format(
            len(text_test_noise_seqs)))

        LOGGER.info("Vectorizing text..")
        X = vectorize(text_test_noise_seqs, self.ctable)
        LOGGER.info("Text vectorized with shape: {}".format(str(X.shape)))

        LOGGER.info("Fixing vectorized sequences..")
        fixed_seqs = self.fix_sequences(
            X, text_test_clipped_seq_ids, text_test_noise_seqs)
        LOGGER.info("Fixing Done.")

        LOGGER.info("Constructing fixed text..")
        # construct fixed text from the fixed_seqs
        fixed_text = self.construct_fixed_text(fixed_seqs)
        LOGGER.info("Fixed text constructed.")

        return fixed_text

    def construct_fixed_text(self, fixed_seqs):
        fixed_text = ''
        for i, s in enumerate(fixed_seqs):
            fixed_text += s
        return fixed_text

    def fix_sequences(self, X, text_test_clipped_seq_ids, text_test_noise_seqs):
        fixed_seqs = []
        for i in range(len(text_test_noise_seqs)):
            # get noisy text
            n_text = text_test_noise_seqs[i][::-1].replace(CONFIG.padding, "")
            x = X[np.array([i])]

            pr = self.fix_sequence(x, i, n_text, text_test_clipped_seq_ids, )

            fixed_seqs.append(pr)
        return fixed_seqs

    def fix_sequence(self, x, seq_ind, n_text, text_test_clipped_seq_ids, ):
        # predict probabilities of the vectorized sequence
        propbs = self.model.predict_proba(x)
        predicted, low_props = self.ctable.decode_proba(propbs[0])
        predicted = predicted.replace(CONFIG.padding, "")
        # use diff log to determine the accepted fixes
        diff_log = get_diff_log(predicted, n_text)
        head_tail_space_log = [d for d in diff_log if
                               d['type'] == 3 and d['chars'][0] == ' ' and d['pos'] == len(predicted) - 1]
        head_tail_space_log = next(iter(head_tail_space_log), None)
        if head_tail_space_log:
            predicted = predicted[:len(predicted) - 1]
            diff_log.remove(head_tail_space_log)
        if len(diff_log) > 1:  # exclude more than one difference
            predicted = n_text
        elif len(diff_log) == 0:  # identical
            predicted = n_text
        else:
            if len(diff_log[0]['chars']) == 1 and diff_log[0]['chars'][0] == ' ':
                predicted = n_text
            elif len(low_props) >= 1:  # only take the confident sequences
                predicted = n_text
            else:
                for d in diff_log:
                    # leave rare characters as is in case of change or deletion
                    if d['type'] == 2:
                        if d['chars'][0] not in self.ctable.chars or d['chars'][0] == '\t':
                            predicted = predicted[:d['pos']] + \
                                d['chars'][0] + predicted[d['pos']:]
                    if d['type'] == 4:
                        if d['chars'][0] not in self.ctable.chars:
                            pr_list = list(predicted)
                            pr_list[d['pos']] = d['chars'][0]
                            predicted = ''.join(pr_list)
                        elif set(d['chars']) == set(['|', ' ']):
                            predicted = n_text
                    # ignore the fixed characters at start and end of excluded sequences
                    if seq_ind in text_test_clipped_seq_ids and d['pos'] == len(predicted) - 1:
                        predicted = n_text
                    elif (seq_ind - 1) in text_test_clipped_seq_ids and d['pos'] == 0:
                        predicted = n_text
        return predicted
