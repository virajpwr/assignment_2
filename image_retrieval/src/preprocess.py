# import necessary libraries

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


class PreprocessText:
    def __init__(self, text):
        self.text = text
        self.update_stop_words =  ['stock', 'images', 'free', 'clip', 'openphoto.be', 
                            'openphoto.meEgret', 'Strikes', 'free', 'photographs', 'for', 'download', 
                            'creative', 'commons', 'licensed', 'images']
        
    def _tokenize(self):
        ''' 
        - tokenise the text using word_tokenize function.
        :return list of tokens
        '''
        tokenizer = TreebankWordTokenizer()
        self.tokens = tokenizer.tokenize(self.text)  
        return self.tokens
    
    def _remove_stopwords(self):
        '''  
        - Remove stop words from the tokens which are the most commonly used words in english text.
        
        :return list of tokens without stopwords
        
        '''
        stop_words = set(stopwords.words('english'))
        stop_words.update(self.update_stop_words)
        return [word for word in self.tokens if word.lower() not in stop_words]
    
    def _convert_num_text(self):
        '''
        - convert numbers to words if the token is a number.
        
        :return list of tokens with numbers converted to words.
        
        '''
        text_num = [num2words(word) if word.isnumeric() and word.isascii() else word for word in self.tokens]
        return text_num
    
    def _normalise_text(self):
        ''' 
        - Normalise the text by converting all the words to lower case and removing punctuations.
        
        :return list of normalised tokens.
        
        '''
        normalised_text = [word.lower() for word in self.tokens if word.isalpha()]
        return normalised_text  
    

    def clean_text(self):
        ''' 
        - Clean the text by tokenising, normalising, removing stopwords and stemming.
        
        :return list of cleaned tokens.
        
        '''
        self._tokenize()
        self.tokens = self._convert_num_text()
        self.tokens = self._normalise_text()
        self.tokens = self._remove_stopwords()
        return self.tokens  
    