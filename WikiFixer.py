
from mle.LanguageModel import LanguageModel
import re


class Dict:
    """special version of python dictionary to help manipulating probabilites  
    where we have list of the keys sorted acording to the order of values
    and if we try to access a value with a key that doesn't exist it return 0 (0 probability)
    """

    def __init__(self, D):
        self.Dict = D
        try:
            self.keys = [k for k in sorted(D, key=D.get, reverse=True)]
        except:
            self.keys = []

        self.count = 0

    def __getitem__(self, key):
        try:
            return self.Dict[key]
        except:
            return 0

    def __iter__(self):
        return self

    def __next__(self):
        count = self.count
        if count >= len(self.keys):
            raise StopIteration
        key = self.keys[count]
        self.count += 1
        return (key, self.Dict[key])


class WikiFixer(object):

    def __init__(self, method="LM", w_th=100):
        self.w_th = w_th
        self.models = []

    def rev(self, s): return s[::-1]

    def check_text(self, text):
        if re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text):
            return 1
        else:
            return 0

    def train_model(self, text, order=4, wi=1, add_k=0, lm={}, rightToLeft=False):
        lm = LanguageModel(order=order, wi=wi, add_k=add_k,
                           lm=lm, rightToLeft=rightToLeft)
        lm.train_char_lm(text)
        clean_lm, clean_lm_cn = self.clean_lm(lm)
        lm.lm = clean_lm
        lm.lm_cn = clean_lm_cn
        self.models.append((lm, 1.0))

    def load_models(self, models=[], lamb=[1.0]):
        if len(models) != len(lamb):
            lamb = [1.0]*len(models)
        for i in range(len(models)):
            lm = None
            lm = LanguageModel()
            lm.load(models[i])
            lb = lamb[i]
            self.models.append((lm, lb))

    def clean_lm(self, lm, cn_thresh=50):
        clean_lm_cn = {}
        clean_lm = {}
        NK = 0.0005
        for key in lm.lm_cn:
            if lm.lm_cn[key] > cn_thresh:
                clean_lm_cn[key] = lm.lm_cn[key]
                clean_lm[key] = []
                for each in lm.lm[key]:
                    if each[1] > NK:
                        clean_lm[key].append(each)
                if not clean_lm[key]:
                    del clean_lm[key]

        return clean_lm, clean_lm_cn

    def fix_text(self, noisy_text, OK=0.8, NK=0.0005, rightToLeft=False, verbose=False, ignoreNum=True, Ignore_SPtext=False):
       
        w_th = self.w_th
        wi = 1
        if rightToLeft:
            noisy_text = self.rev(noisy_text)
        text_data = list(noisy_text)
        order = self.models[0][0].order
        lm_cn = self.models[0][0].lm_cn
        total = len(text_data)
        Total = total
        total -= (order+2*self.models[0][0].wi)
        errors = {"swap": 0, "deletion": 0, "insertion": 0, "change": 0}
        errors_pos = []
        i = 0
        ln = order
        lns = []
        #replace_pattern
        o_word = list(text_data[:order])    # original word
        f_word = list(text_data[:order])      # fixed word
        # pervious original word (in case, to be used when model fixes space between two words)
        l_oword = ""
        l_fword = ""     # pervious fixed word
        label = 0
        pos_change = 0
        replace = []
        labels = []
        f_pos = [0, 0]
        while i < total:
            cr1 = ""
            cr2 = ""
            z = True
            char2 = ""
            k = ""

            error_type = -1
            history, char = text_data[i:i+order], text_data[i+order]
            p1 = i+order

            hist, ch = ["".join(history)], str(char)

            if ch == "\n":
                lns.append(ln)
                ln = 0
            else:
                ln += 1
            f_word.append(str(char))
            o_word.append(str(char))
            # and "\n" not in "".join(char) and "\n" not in "".join(history):

            if not ignoreNum or (not("".join(char).isnumeric()) and not("".join(history).isnumeric()) ):
                a = 0 
                u = 0
                # for model in self.models:
                for u in range(len(self.models)):
                    probs_a = Dict(
                        self.models[u][0].print_probs(''.join(history)))
                    a += ((probs_a[char])*(self.models[u][1]))
                    u += 1
                probs_a = Dict(self.models[0][0].print_probs(''.join(history)))
                try:
                    cn = lm_cn["".join(history)]
                except:
                    cn = 0
                if probs_a and cn >= w_th and not (Ignore_SPtext and self.check_text(str("".join(o_word)))):
                    if a < NK:
                        history, char = text_data[i:i +
                                                  order], text_data[i+order+1]
                        probs_b = Dict(
                            self.models[0][0].print_probs(''.join(history)))
                        b = probs_b[char]
                        if b > OK:
                            history, char = text_data[i+1:i + order] + \
                                [text_data[i+order+1]], text_data[i+order]
                            probs_c = Dict(
                                self.models[0][0].print_probs(''.join(history)))
                            c = probs_c[char]
                            # swap error
                            if c > OK and not text_data[i+order+1].isnumeric():
                                char1 = text_data[i+order]
                                char2 = text_data[i+order+1]
                                cr2 = str(char2)
                                cr1 = str(char1)

                                text_data[i+order] = char2
                                text_data[i+order+1] = char1
                                error_type = 1

                                #f_word[-1] = char2
                                f_word.insert(-1, str(cr2))
                                #f_word.append(str(cr1))

                                #o.word(char1)
                                #o_word.append(str(cr1))
                                o_word.append(str(cr2))

                                i += 1
                                z = False
                            if z:
                                history, char = text_data[i+1:i + order] + \
                                    [text_data[i+order+1]], text_data[i+order+2]

                                probs_g = Dict(
                                    self.models[0][0].print_probs(''.join(history)))
                                g = probs_g[char]
                                if g > OK:  # insertion error
                                    del text_data[i+order]
                                    cr1 = text_data[i+order]
                                    cr2 = ""
                                    del f_word[-1]
                                    #o_word.append(text_data[i+order])
                                    error_type = 2
                                    total -= 1
                                    i -= 1
                                    z = False
                        if z:
                            for k, d in probs_a:
                                if d < OK:
                                    break
                                else:
                                    history, char = text_data[i+1:i +
                                                              order]+[k], text_data[i+order]
                                    probs_e = Dict(
                                        self.models[0][0].print_probs(''.join(history)))
                                    e = probs_e[char]

                                    history, char = text_data[i+1:i +
                                                              order]+[k], text_data[i+order+1]
                                    probs_f = Dict(
                                        self.models[0][0].print_probs(''.join(history)))
                                    f = probs_f[char]

                                    if e > f and e > OK and not k.isnumeric():  # deletion error
                                        text_data.insert(i+order, k)
                                        cr2 = k
                                        cr1 = ch
                                        f_word.insert(-1, k)
                                        error_type = 3
                                        total += 1
                                        i += 1
                                        break
                                    # change error
                                    elif f > OK and not("".join(k).isnumeric()):
                                        f_word[-1] = k
                                        #o_word.append(text_data[i+order])
                                        text_data[i+order] = k
                                        cr2 = k
                                        cr1 = ch
                                        error_type = 4
                                        break
                                    elif f < OK and e < OK:
                                        break
                    else:
                        text_data[i+order] = ch
                        cr1 = ch
                        cr2 = ch

                else:
                    text_data[i+order] = ch
                    cr1 = ch
                    cr2 = ch

            if cr1 in [" ", "\n"] and cr2 in [" ", "\n"]:  # if there is a word separator
                l_oword = str("".join(o_word))
                l_fword = str("".join(f_word))
                fword = str("".join(f_word))
                oword = str("".join(o_word))
                if rightToLeft:
                    fword = self.rev(fword)
                    oword = self.rev(oword)
                f_pos[1] = i+order+1
                replace.append(
                    {"f_word": fword, "o_word": oword, "f_pos": list(f_pos)})
                f_pos[0] = i+order+1

                o_word = []
                f_word = []
                label += 1
            R = ["".join(text_data[i+order+1:i+2*order+1])]
            Rd = ["".join(text_data[i+order:i+2*order])]
            if rightToLeft:
                hist = [self.rev(hist[0])]
                R = [self.rev(R[0])]
                Rd = [self.rev(Rd[0])]

            if error_type in [1, 2, 3, 4]:
                if error_type == 1:
                    errors["swap"] += 1
                    errors_pos.append(
                        {"t": "1", "p1": p1, "p2": p1+1, "history": hist, "char": ch, "char2": char2, "k": k,
                         "L": hist, "R": R})
                elif error_type == 2:
                    pos_change += 1
                    errors["insertion"] += 1
                    errors_pos.append(
                        {"t": "2", "p1": p1, "p2": p1, "history": hist, "char": ch, "char2": char2, "k": k,
                         "L": hist, "R": R})
                elif error_type == 3:
                    pos_change -= 1
                    errors["deletion"] += 1
                    errors_pos.append(
                        {"t": "3", "p1": p1, "p2": p1, "history": hist, "char": ch, "char2": char2, "k": k,
                         "L": hist, "R": Rd})
                elif error_type == 4:
                    errors["change"] += 1
                    errors_pos.append(
                        {"t": "4", "p1": p1, "p2": p1, "history": hist, "char": ch, "char2": char2, "k": k,
                         "L": hist, "R": R})
                labels.append(label)

            i += 1

        ln += 2
        lns.append(ln)
        fword = str("".join(f_word))
        oword = str("".join(o_word))
        if rightToLeft:
            fword = self.rev(fword)
            oword = self.rev(oword)
        f_pos[1] = i+order+1
        replace.append(
            {"f_word": fword, "o_word": oword, "f_pos": list(f_pos)})

        replaces = []
        h = -1

        for label in labels:
            if label != h:  # h for history, so we don't replicate words in case it contains two errors
                replaces.append(replace[label])
            h = label
        text = ''.join(text_data)
        errors_text = ["0"]*len(text)

        #print(Total)
        #Total += pos_change-1
        total += (order+2*self.models[0][0].wi)

        #print(Total)
        for i in range(len(errors_pos)):

            if rightToLeft:
                #ln=lns[i]

                p1 = errors_pos[i]["p1"]
                p2 = errors_pos[i]["p2"]
                if errors_pos[i]["t"] == "1":
                    p1, p2 = p2, p1
                    errors_pos[i]["p1"] = total-p1-1
                    errors_pos[i]["p2"] = total-p2-1
                    errors_pos[i]["char"], errors_pos[i]["char2"] = errors_pos[i]["char2"], errors_pos[i]["char"]

                elif errors_pos[i]["t"] == "2":

                    errors_pos[i]["p1"] = total-p1
                    errors_pos[i]["p2"] = total-p2

                elif errors_pos[i]["t"] == "3":

                    errors_pos[i]["p1"] = total-p1-1
                    errors_pos[i]["p2"] = total-p2-1

                elif errors_pos[i]["t"] == "4":
                    errors_pos[i]["p1"] = total-p1-1
                    errors_pos[i]["p2"] = total-p2-1

            error = errors_pos[i]
            p1 = error["p1"]
            p2 = error["p2"]+1
            t = error["t"]
            y = p2-p1
            errors_text[p1:p2] = [t]*y


        #if rightToLeft:
        #    errors_pos.reverse()

        errors_text = ''.join(errors_text)

        if rightToLeft:
            text = self.rev(text)
            errors_text = self.rev(errors_text)
            errors_pos = self.rev(errors_pos)
            replaces = self.rev(replaces)

        if verbose:
            return text, errors, errors_pos, errors_text, replaces
        else:
            return text
