import streamlit as st 
import pandas as pd 
import numpy as np
import yaml
import pickle


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


import math
from functools import lru_cache
import math
import operator
from tqdm import tqdm
import itertools
from collections import Counter
from dataclasses import dataclass
from math import log
import pickle

from src import preprocess, invertedindex

import streamlit as st 
import pandas as pd 
import numpy as np
import yaml
import pickle


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


import math
from functools import lru_cache
import math
import operator
from tqdm import tqdm
import itertools
from collections import Counter
from dataclasses import dataclass
from math import log
import pickle

from src import preprocess, invertedindex

class RunModel:
    def __init__(self, index, mapping):
        self._document_collection = index.documents_collection
        self._index = index.get_index
        self._mapping = mapping
        
    def rank(self, qid, query):
        '''
        - Clean queries and get the relevant documents for each query word.
        
        :param qid: query id
        :param query: query text
        
        :return: dictionary with docid and score as key-value pairs.
        '''
        query_words = self._clean_query_text(query)
        docs = self._relevant_docs(qid, query_words)

        scores = {}
        for docid in docs:
            scores[docid] = self._score_docs(docid, query_words)

        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranks = {docid: score for docid, score in ranking}
        return ranks

    def _clean_query_text(self, query):
        clean_query_text = [word for word in preprocess.PreprocessText(query).clean_text()]
        return clean_query_text

    def _relevant_docs(self, qid, query_words):
        ''' 
        - Helper function to get the relevant documents for each query word.
        
        :param qid: query id
        :param query_words: list of cleaned query words
        
        :return: list of relevant documents for each query word.
        '''
        relevant_docs_set = set() # set to avoid duplicates
        # get the relevant documents for each query word
        for word in query_words:  # iterate over the query words
            if word in self._index: # check if the word is in the index
                for docid in self._index[word].postings: # get the docids for each query word
                    if docid in self._mapping[qid]: # check if the docid is relevant to the query
                        relevant_docs_set.add(docid) # add the docid to the set
        return list(relevant_docs_set)

class VSM(RunModel):
    def __init__(self, index, mapping):
        super().__init__(index, mapping)
        self._vocab = index.vocab
        self._collection_length = index.document_length
        self._vocab_count = index.vocab_count

    def _score_docs(self, docid: int, query_words) -> float:
        vocab = list(set(self._document_collection[docid]))
        vocab_count = len(vocab)

        doc_vector = np.zeros(vocab_count)
        for i, word in enumerate(vocab):
            doc_vector[i] = self._index[word].get_posting(docid).tfidf

        query_vector = np.zeros(vocab_count)
        counter = Counter(query_words)
        max_freq = counter.most_common(1)[0][1]
        for word in query_words:
            if word not in self._index:
                continue
            tf = (0.5 + (0.5 * (counter[word] / max_freq))) # 
            idf = log(self._collection_length / self._index[word].document_frequency)
            tfidf = tf * idf
            if word in vocab:
                wordidx = vocab.index(word)
                query_vector[wordidx] = tfidf
            else:
                query_vector = np.append(query_vector, tfidf)
                doc_vector = np.append(doc_vector, 0)
                
        cosine_similarity = np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
        return cosine_similarity


def mapping(query, doc):    
    ''' 
    - Map each query id with all the document ids.
    
    :param query: dictionary with queryid and query as key-value pair.
    :param doc: dictionary with docid and text as key-value pairs.
    
    :return: dictionary with queryid as key and list of document ids as value.
    '''    
    mappings = {}
    for key, value in query.items():
        for k, v in doc.items():
            doc_keys = list(doc.keys())
            doc_keys = [int(i) for i in doc_keys]
            mappings[int(key)] = doc_keys 
    return mappings

def generate_index(doc_collection):
    index = invertedindex.InvertedIndexDocs(doc_collection)
    index.create_index()
    return index


# Set the app title 
st.title('Assignment 2') 



def search_query(input_query, docs, inv):
    map_doc_queries = mapping(input_query, docs)
    model = VSM(inv, map_doc_queries)

    # loop through each query and perform ranking
    ranking = {qid: model.rank(qid, query) for qid, query in clean_query.items()}

    filter_rank = {k: v for k, v in ranking[1].items() if v > 0.1}
    result = defaultdict(dict)
    for key, value in filter_rank.items():
        result[key]['score'] = value
        result[key]['url'] = urls[key]
        result[key]['image_name'] = image_data[key]['image_name']
        result[key]['photographer'] = image_data[key]['photographer']
        result[key]['image_id'] = key
    return result
    

genre = st.sidebar.radio(
    "Please select type",
    ('Image Search', 'Search Query Examples'))

if genre == 'Image Search':
    st.subheader(
    '**👈 Please select the radio button on the sidebar for search query examples**')
    st.subheader(
        'Image search engine')
    
    st.link_button("Image source", "https://openphoto.net")
    with open(r'./data/image_data.pkl', 'rb') as f:
        image_data = pickle.load(f)
        

    urls= {image_id: image_data[image_id]['image_source'] for image_id in image_data}
    documents= {image_id: image_data[image_id]['image_details'] for image_id in image_data}
    
    inv_index = generate_index(documents)
    
    with open(r'./data/stopwords.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    stopwords = set(config['stop words'])

    widgetuser_input = st.text_input('Enter a search query') 
    
    
    pattern = r'\b[A-Za-z]+\b'
    query_words = re.findall(pattern, widgetuser_input)

    # lowercase 
    query_words = [word.lower() for word in query_words]
    # remove stopwords
    query_words = [word for word in query_words if word not in stopwords]

    query = {1:query_words}
    clean_query = {key: ' '.join(value) for key, value in query.items()}

    ranking_dict = search_query(clean_query, documents, inv_index)
    
    for key, value in ranking_dict.items():
        st.write(f'Image ID {value["image_id"]} | Image Name {value["image_name"]} | Similarity Score: {value["score"]} | Photographer: {value["photographer"]}') 
        st.image(value["url"], caption="searched image")


if genre == 'Search Query Examples':
    st.subheader(
    'Search query examples')
    
    multi = '''
            white horse in a field

            african capybara
            
            images by Michael Jastremski 
            
            strawberry mansion bridge
            
            philadelphia river 

            croatian wild duck

            cute rabbit

            green tomatoes on vine

            daffodil yellow petals

            Scuba diving with coral reef

            national park in croatia

            japanese temple

            Tokyo Disney Land
            
            westminster cathederal sign

            London Tube 

            eiffel tower at night

            tulipe flower
    '''
    st.markdown(multi)

