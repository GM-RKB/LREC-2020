from path import get_path
from mli.LanguageModel import LanguageModel

filename = get_path("models/model1-0.json")
outfile = get_path("models/cleaned/model1-0.json")
l = LanguageModel()
l.load(filename)
clean_lm_cn={}
clean_lm={}

cn_thresh = 50
NK=0.0005

for key in l.lm_cn:
    if l.lm_cn[key]>cn_thresh:
        clean_lm_cn[key] = l.lm_cn[key]
        clean_lm[key]=[]
        for each in l.lm[key]:
            if each[1] > NK:
                clean_lm[key].append(each)
        if not clean_lm[key]:
            del clean_lm[key]


l.lm=clean_lm
l.lm_cn=clean_lm_cn

l.save(outfile)

