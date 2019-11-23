import numpy as np
from numpy import zeros as np_zeros
import json

from model_config import CONFIG


def get_allowed_chars():
    """
    Get allowed characters
    :return:
    """
    with open(CONFIG.ALLOWED_CHARS_FILE_NAME, 'r') as fp:
        return [ch for _, ch in json.load(fp).items()]


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    @property
    def size(self):
        """The number of chars"""
        return len(self.chars)

    def encode(self, C, maxlen):
        """Encode as one-hot"""
        X = np_zeros((maxlen, len(self.chars)), dtype=np.bool)
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        """Decode from one-hot"""
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X if x)

    def decode_proba(self, P):
        """Decode from the model probabilities"""
        c = {}
        for i in range(CONFIG.max_input_len):
            if max(P[i]) < CONFIG.allowed_threshold:
                c[i] = max(P[i])
        P = P.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in P if x), c


def get_char_table():
    allowed_chars = get_allowed_chars()
    return CharacterTable(allowed_chars)


def vectorize(noisy_texts, ctable):
    """vectorize the data as numpy arrays"""
    len_of_noisy_texts = len(noisy_texts)
    X = np_zeros((len_of_noisy_texts, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    for i in range(len_of_noisy_texts):
        sentence = noisy_texts[i]
        for j, c in enumerate(sentence):
            try:
                X[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass  # Padding
    return X
