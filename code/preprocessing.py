#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import spacy
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import stanza
from stanza_batch import batch
from stanza.models.common.doc import Document
import itertools
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
stop_words = set(stopwords.words('english'))


# remove copy right information (c)
def copyright(doc):
    for i in re.findall('\(C\).+',doc): 
        doc=doc.replace(i,'')
    return doc

# merge abstract title and keywords
def merge(x):
    if pd.notnull(x.AB):
        if pd.notnull(x.Title) and pd.notnull(x.Keywords):
            return x.AB+' '+ x.Title +'. '+x.Keywords+' '
        else:
            return x.AB+' '+x.Title+ ' '
    else:
        np.nan
        
#change uk to us spelling
#see http://www.tysto.com/uk-us-spelling-list.html
uk2us=pd.read_excel('uk_us.xlsx').drop(425).reset_index(drop=True) 
k2s = dict(zip(uk2us.UK, uk2us.US))

#one to one replacement based on a dictionary
def replace_all(text, mydict):
    for gb, us in mydict.items():
        text=text.replace(gb, us)
    return text

''' preprocess texts with Spacy pipeline: custom tokenizer and lemmatizer '''

# customize tokenizer for hypen connector
def custom_tokenizer(nlp):
    infix_re = re.compile(r'~')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    return Tokenizer(nlp.vocab,infix_finditer=infix_re.finditer,prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search)
# filter out punctuation, space and stopwords 
def token_filter(token):
    return not (token.is_punct | token.is_space | token.is_stop)

def preprocess_doc(texts):
    # rule-based lemmatizer exceptions for central words such as media, data, learning, embedding
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.tokenizer = custom_tokenizer(nlp)
    nlp.get_pipe('lemmatizer').lookups.get_table("lemma_exc")["noun"]["data"] = ["data"]
    nlp.get_pipe('lemmatizer').lookups.get_table("lemma_exc")["noun"]["media"] = ["media"]
    nlp.get_pipe('lemmatizer').lookups.get_table("lemma_exc")["verb"]["learning"] = ["learning"]
    nlp.get_pipe('lemmatizer').lookups.get_table("lemma_exc")["verb"]["embedding"] = ["embedding"]
    assert nlp("data")[0].lemma_ == "data"
    assert nlp("media")[0].lemma_ == "media"
    texts=[replace_all(doc.lower(), k2s) for doc in texts]
    filtered_tokens = []
    for doc in tqdm(nlp.pipe(texts,batch_size=50)):
        tokens = [token.text for token in doc if token_filter(token)]
        tokens = [token.lemma_ for token in doc if token_filter(token)]
        filtered_tokens.append(tokens)
    return filtered_tokens


# altenative: stanza pipeline on a list of documents 
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')
def preprocess_doc_stanza(texts):
    # Wrap each document with a stanza.Document object
    stanza_documents: List[Document] = []
    texts=[replace_all(doc.lower(), k2s) for doc in texts]
    final_doc=[]
    for document in tqdm(batch(texts, nlp, batch_size=50)): #in batches 
        stanza_documents.append(document)
    for i in range(len(stanza_documents)): #process in loops 
        final_tokens = []
        for sent in stanza_documents[i].sentences:
            for each in sent.words:   
                each=re.sub(r'[^\w\s-]','',each.lemma) #lemmatization and remove punctuations
                if each:
                    final_tokens.append(each)  
        final_doc.append([each for each in simple_coll(final_tokens) if each not in stop_words]) # remove stopwords
    assert len(final_doc)==len(texts)
    return final_doc

# get possible the combination of terms with connector "-" caused by machine collocation
def phrase_combination(a):
    ls=[]
    if len(a.split(' '))>1:
        a=' '+a+' '
        for i in range(len(a.split(' '))):
            if i<len(a.split(' '))-1:
                ls.append([a.split(' ')[i]+'-',a.split(' ')[i]+' '])
            else:
                ls.append([a.split(' ')[i]])
        ls=[''.join(list(i)) for i in list(itertools.product(*ls))]   
    else:
        ls=[a]
    return ls




