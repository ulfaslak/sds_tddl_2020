## Lexicon Mining
## Collection and Rough implementation of lexical based sentiment and aspect analysis
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import re,os
import nltk
if not os.path.isdir('lexicon_functions'):
    lexicon_path = input().strip('/')
else:
    lexicon_path = 'lexicon_functions'
## Load precompiled functions and dependencies
w2subj,subjectivity_types = pickle.load(open(lexicon_path+'/subjectivity_score.pkl','rb'))
def get_subjectivity(doc, tokenizer=nltk.word_tokenize,agg='mean'):
    if type(doc)==str:
        doc = tokenizer(doc)
    assert type(doc)==list,"please input either a list or a string"
    if len(doc)==0:
        return {}
    matches = Counter()
    for w in doc:
        w = w.lower()
        if w in w2subj:
            matches[w2subj[w]]+=1

    if len(matches)==0:
        return {typ:0 for typ in subjectivity_types}
    scores = pd.Series(np.array([matches[typ] for typ in subjectivity_types]),index=subjectivity_types)
    if agg=='mean':
        scores =  scores/len(doc)
    elif agg =='abs':
        scores = scores
    else:
        scores =  agg(scores)
    return dict(scores)



class2re,string_test = pickle.load(open(lexicon_path+'/text2arg.pkl','rb'))
def text2argfeatures(text):
    d = {}
    for name,regex in class2re.items():
        d[name] = len(regex.findall(text))
    return d


w2scores = pickle.load(open(lexicon_path+'/vad_score.pkl','rb'))
def get_vad_score(doc,tokenizer=nltk.word_tokenize,agg='mean'):
    if type(doc)==str:
        doc = tokenizer(doc)
    assert type(doc)==list,"please input either a list or a string"

    matches = []
    for w in doc:
        w = w.lower()
        if w in w2scores:
            matches.append(w2scores[w])
    if len(matches)==0:
        return {'arousal':np.nan,'dominance':np.nan,'valence':np.nan}
    scores = pd.DataFrame(matches)
    if agg=='mean':
        scores =  scores.mean()
    elif agg=='max':
        scores =  scores.max()
    else:
        scores =  agg(scores)
    return dict(scores)

w2affects = pickle.load(open(lexicon_path+'/ail_score.pkl','rb'))
def get_affect_intensity_score(doc,tokenizer=nltk.word_tokenize,agg='mean'):
    if type(doc)==str:
        doc = tokenizer(doc)
    assert type(doc)==list,"please input either a list or a string"
    matches = []
    for w in doc:
        w = w.lower()
        if w in w2affects:
            matches.append(w2affects[w])
    if len(matches)==0:
        return {'anger':np.nan,'joy':np.nan,'sadness':np.nan,'fear':np.nan}
    scores = pd.DataFrame(matches)
    if agg=='mean':
        scores = scores.mean()
    elif agg=='max':
        scores =  scores.max()
    else:
        scores = agg(scores)
    return dict(scores)

w2conglomerate,conglomerate_cols = pickle.load(open(lexicon_path+'/conglomerate.pkl','rb'))
def get_conglomerate_scores(doc,tokenizer=nltk.word_tokenize,agg='mean'):
    if type(doc)==str:
        doc = tokenizer(doc)
    assert type(doc)==list,"please input either a list or a string"
    matches = []
    for w in doc:
        w = w.lower()
        if w in w2conglomerate:
            matches.append(dict(list(zip(conglomerate_cols,w2conglomerate[w]))))
    if len(matches)==0:
        return dict(list(zip(conglomerate_cols,[np.nan]*len(conglomerate_cols))))
    scores = pd.DataFrame(matches)
    if agg=='mean':
        scores = scores.mean()
    elif agg =='max':
        scores  = scores.max()
    else:
        assert hasattr(agg,'__call__'),'"agg" should be a function if not "mean" or "max"'
        scores = agg(scores)
    return dict(scores)

w2happy = pickle.load(open(lexicon_path+'/happiness.pkl','rb'))
def get_happiness(doc,tokenizer=nltk.word_tokenize,agg='sum'):
    if type(doc)==str:
        doc = tokenizer(doc)
    assert type(doc)==list, 'please input string or list'

    scores = []
    for w in doc:
        if w in w2happy:
            scores.append(w2happy[w])
    score = np.mean(scores)
    return {'happiness':score}
## Define builtin methods
### LIU
from nltk.corpus import opinion_lexicon
positive_w = set(opinion_lexicon.positive())
negative_w = set(opinion_lexicon.negative())
def get_pos_neg_liu(doc,tokenizer=nltk.word_tokenize,agg='sum'):
    if type(doc)==str:
        doc = [i.lower() for i in tokenizer(doc)]
    assert type(doc)==list, 'input has to be either string or list'
    if len(doc)==0:
        return {'positive_count':np.nan,'negative_count':np.nan}

    d = {'positive_count':count_words(doc,positive_w),
        'negative_count':count_words(doc,negative_w)}
    if agg=='sum':
        return d
    elif agg=='mean':
        return {key:val/len(doc) for key,val in d.items()}

def count_words(doc,s):
    c = Counter(doc)
    return sum([c[i] for i in s])

## AFINN
from afinn import Afinn
afinn = Afinn(emoticons=True)
def get_afinn(text):
    if type(text)==list:
        text = ' '.join(text)
    return {'afinn':afinn.score(text)}

## Vader
import nltk.sentiment
vader = nltk.sentiment.vader.SentimentIntensityAnalyzer()
##########################3
#### WRAP them all in one big function


name2func = {'liu':get_pos_neg_liu,
             'conglomerate':get_conglomerate_scores,
             'affect_intensity':get_affect_intensity_score,
             'vad':get_vad_score,
             'subjectivity':get_subjectivity,
             'hedometer':get_happiness
            }
textbased_funcs = {'vader':vader.polarity_scores,
             'afinn':get_afinn,
                  'argumentation':text2argfeatures}

def lexical_mining(text,tokenizer = nltk.word_tokenize,agg = {}):
    if type(text)==str:
        doc = tokenizer(text)
    if type(text)==np.nan:
        return np.nan
    d = {}
    for name,func in textbased_funcs.items():
        temp_d = {'%s_%s'%(name,key):val for key,val in func(text).items()}
        d.update(temp_d)
    for name,func in name2func.items():
        temp_d = {'%s_%s'%(name,key):val for key,val in func(doc).items()}
        d.update(temp_d)
    return pd.Series(d)
