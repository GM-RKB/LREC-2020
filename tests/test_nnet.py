import unittest

from nnet import data_processing, data_vectorization
from model_config import CONFIG
import numpy as np


class TestWikiFixerNNet(unittest.TestCase):
    test_text = "'''See:''' [[User]], [[Behavior]], [[Click-Through Data]], [[User Interface]].\n----\n\n__NOTOC__\n[[Category:Stub]]"

    def test_split_text(self):
        split_3 = ["'''See:''' [[User]], [[Behavior]], ",
                   '[[Click-Through Data]], [[User ',
                   'Interface]].\n----\n\n__NOTOC__\n[[Category:Stub]]']
        split_4 = ["'''See:''' [[User]], [[Behavior]], [[Click-Through ",
                   'Data]], [[User Interface]].\n----\n\n__NOTOC__\n[[Category:Stub]]']
        split_padding = [
            "'''See:''' [[User]], [[Behavior]], [[Click-Through Data]], [[User Interface]].\n----\n\n__NOTOC__\n[[Category:Stub]]"]

        self.assertEqual(data_processing.split_text(self.test_text, 3, 3), split_3)
        self.assertEqual(data_processing.split_text(self.test_text, 4, 4), split_4)
        self.assertEqual(data_processing.split_text(self.test_text, 4, 4, CONFIG.padding), split_padding)

    def test_convert_text_to_sequences(self):
        # assume sequence length = 40, change here if different length
        # CONFIG.max_input_len = 40
        seq_space = (["'''See:''' [[User]], [[Behavior]], ",
                      '[[Click-Through Data]], [[User ',
                      'Interface]].\n----\n\n__NOTOC__\n[[Category:',
                      'Stub]]'],
                     [2])
        seq_no_space = (["'''See:''' [[User]], [[Behavior]], [[Cli",
                         'ck-Through Data]], [[User Interface]].\n-',
                         '---\n\n__NOTOC__\n[[Category:Stub]]'],
                        [0, 1])
        self.assertEqual(data_processing.convert_text_to_sequences(self.test_text), seq_space)
        self.assertEqual(data_processing.convert_text_to_sequences(self.test_text, CONFIG.padding), seq_no_space)

    def test_padding_invert(self):
        test_padding_invert = "'''See:''' [[User]], [[Behavior]], "
        padding_res = "'''See:''' [[User]], [[Behavior]], 🔳🔳🔳🔳🔳"
        invert_res = " ,]]roivaheB[[ ,]]resU[[ ''':eeS'''"

        self.assertEqual(data_processing.mask_digits_complete_seq(test_padding_invert), padding_res)
        self.assertEqual(data_processing.invert_seq(test_padding_invert), invert_res)

    def test_vectorize(self):
        CONFIG.max_input_len = 3
        allowed_chars = ['a', 'b', 'c']
        noise_seqs = ["abc"]
        vectorize_res = np.array([[[True, False, False],
                                   [False, True, False],
                                   [False, False, True]]])
        ctable = data_vectorization.CharacterTable(allowed_chars)

        self.assertTrue(np.allclose(data_vectorization.vectorize(noise_seqs, ctable), vectorize_res))
