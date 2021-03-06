from WikiFixerMLE import WikiFixer
from tests.path import get_path
from WikiFixerNNet import WikiFixerNNet
import enchant
import jamspell




class jspell(object):
        def __init__(self):
                self.corrector = jamspell.TSpellCorrector()
                self.corrector.LoadLangModel('/mnt/efs/data/en.bin')

        def fix_text(self, text):
                out = []
                for line in text.splitlines():
                        out.append(self.corrector.FixFragment(line))
                return "\n".join(out)


class Enchant(object):
        def __init__(self):
                self.corrector = enchant.Dict("en_US")

        def fix_text(self, text):
                out = []
                for line in text.splitlines():
                        newline=[]
                        for word in line.split():
                                if self.corrector.check(word):
                                        newline.append(word)
                                else:
                                        try:
                                                newline.append(self.corrector.suggest(word)[0])
                                        except:
                                                newline.append(word)
                                
                        out.append(" ".join(newline))
                return "\n".join(out)

def get_config(Model,datafile):
        if Model == "mle":
                fixer = WikiFixer()
                models = [get_path("/mnt/efs/models/model6-0.json")]
                lamb = [1.0]
                fixer.load_models(models=models, lamb=lamb)
        elif Model == "jspell":
                fixer = jspell()
        elif Model.lower() == "enchant":
                fixer=Enchant()
        else:
                fixer = WikiFixerNNet()
                fixer.load_model()

        
        config = {"fixer": fixer, #load fixer 
                  "sample_size": None, #set number of test samples
                  "i1":0, #set range from i1
                  "i2":100, # to i2
                  "data_file":datafile, #load data file
                  "k": None # use instances with label k=1 (not included in training)
                  }
        
        return config
