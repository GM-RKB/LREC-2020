from model_config import CONFIG


def split_text(text, s_len, s_step, s_char=' '):
    """Split the text into s_len phrases, sliding with s_step by s_char"""
    lines = []
    i = 0
    words = text.split(s_char)
    w_len = len(words) - 1
    while True:
        if i + (s_len - 1) < w_len:
            lines.append(s_char.join(words[i:i + s_len]) + s_char)
        else:
            lines.append(s_char.join(words[i:]))
            break
        i += s_step
    return lines


def convert_text_to_sequences(text, split_char=' '):
    '''
    Convert text into sequences adjust to the network max seq length
    :param text: text needed to be converted
    :param split_char: split char with space as default
    :return: converted sequences + index where unnatural cut performed
    '''
    clipped_seq_ids = []
    seqs = []

    while text:
        if len(text) <= CONFIG.max_input_len:
            seqs.append(text)
            break
        else:
            split_location = text[:CONFIG.max_input_len].rfind(split_char)
            if split_location == -1 or split_location >= CONFIG.max_input_len:
                clipped_seq_ids.append(len(seqs))
                seqs.append(text[:CONFIG.max_input_len])
                text = text[CONFIG.max_input_len:]
            else:
                seqs.append(text[:split_location + 1])
                text = text[split_location + 1:]
    return seqs, clipped_seq_ids


def mask_digits_complete_seq(text):
    text += CONFIG.padding * (CONFIG.max_input_len - len(text))
    return text


def invert_seq(noisy_text):
    return noisy_text[::-1] if CONFIG.inverted else noisy_text


def process_text(text):
    """
    Use the functions in this module to convert text to sequences ready to be vectorized
    :param text: the text needed to be fixed
    :return: the converted text sequences
    """
    split_text_noise = split_text(text, 4, 4)

    text_test_noise_seqs = []
    text_test_clipped_seq_ids = []
    i = 0
    for xt in split_text_noise:
        ts, xs = convert_text_to_sequences(xt)
        text_test_noise_seqs.extend(ts)
        for s_ind in xs:
            text_test_clipped_seq_ids.append(i + s_ind)
        i += len(ts)

    for i, seq in enumerate(text_test_noise_seqs):
        text_test_noise_seqs[i] = invert_seq(mask_digits_complete_seq(seq))

    return text_test_noise_seqs, text_test_clipped_seq_ids
