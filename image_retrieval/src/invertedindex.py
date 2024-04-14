from src import preprocess
from dataclasses import dataclass
# import necessary libraries

import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm
 
# utilities
from collections import defaultdict
import re
import numpy as np
import pandas as pd
from copy import deepcopy
import os
from itertools import chain
import operator
# nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import (TreebankWordTokenizer, TreebankWordDetokenizer)
from string import punctuation
from nltk import PorterStemmer
from collections import Counter

import math
from functools import lru_cache
import numpy as np
import math
import operator
from tqdm import tqdm
import itertools
from collections import Counter
from dataclasses import dataclass
from math import log
import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass  
class Posting:
    freq: int = 1
    tfidf: float = 0.0
    
class InvertedIndexDocs:
    def __init__(self, documents):
        self._documents = self._clean_data(documents)
        self._indexdocs = dict()
        
    def _clean_data(self, document_list):
        ''' 
        - Clean the text data by removing stopwords, normalising, stemming and converting numbers to words.
        
        :param document_list: dictionary with docid and text as key-value pairs.
        
        :return: dictionary with docid and cleaned text as key-value pairs.
        '''
        clean_docs = {doc_id: preprocess.PreprocessText(doc_text).clean_text() for doc_id, doc_text in document_list.items()}
        return clean_docs
    
    def create_index(self):
        '''
        - Create an inverted index with docid and posting term frequency and inverse document frequency.
        
        '''
        self._index_docs()
        self._tfidf_docs()
        
    def _index_docs(self):
        for docid, docs in self._documents.items():
            for token in docs:
                if token not in self._indexdocs:
                    self._indexdocs[token] = InvertList()
                self._indexdocs[token].add_posting_doc(docid)
    
    def _tfidf_docs(self):
        for token, inv_index in self._indexdocs.items():
            for _, posting in inv_index.postings.items():
                term_freq = posting.freq
                idf = log(self.document_length / self._indexdocs[token].document_frequency)
                posting.tfidf = term_freq * idf
                
    @property
    def get_index(self):
        indexed_doc = {k:v for k,v in sorted(self._indexdocs.items())}
        return indexed_doc
    
    @property
    def documents_collection(self):
        return self._documents

    @property
    def words(self):
        ''' 
        - Get all the words from the cleaned documents.
        '''
        return list(itertools.chain.from_iterable(self._documents.values()))

    @property
    def vocab(self):
        ''' 
        - Get the vocabulary from the cleaned documents.
        '''
        return sorted(self._indexdocs)

    @property
    def document_length(self):
        ''' 
        - Get the length of the documents.
        '''
        return len(self._documents)

    @property
    def word_count(self):
        ''' 
        - Get the total number of words in the documents.
        '''
        return len(self.words)

    @property
    def vocab_count(self):
        ''' 
        - Get the total number of unique words in the documents.
        '''
        return len(self._indexdocs)

    @property
    def avg_doc_length(self):
        ''' 
        - Get the average document length.
        '''
        return self.word_count / self.document_length

    @property
    def counter(self):
        ''' 
        - Get the frequency of each word in the documents.
        '''
        return Counter(self.words)
    
class InvertList:
    def __init__(self):
        self._postings = dict()

    def add_posting_doc(self, docid):
        if self.contains_posting_data(docid):
            return self.update_posting(docid)
        posting = Posting()
        self._postings[docid] = posting

    def update_posting(self, docid):
        if not self.contains_posting_data(docid):
            return self.add_posting_doc(docid)
        posting = self._postings[docid]
        posting.freq += 1

    def get_posting(self, docid):
        return self._postings[docid]

    def contains_posting_data(self, docid):
        return docid in self._postings

    @property
    def postings(self):
        return self._postings

    @property
    def document_frequency(self):
        return len(self._postings)