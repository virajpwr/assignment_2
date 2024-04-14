
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
        clean_query_text = [word for word in PreprocessText(query).clean_text()]
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