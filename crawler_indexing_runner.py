from src import crawler, indexing, invertedindex, loadweights, preprocess, annotate, setting, utility
import json
import os
from tqdm import tqdm
import pandas as pd 
import pickle
import numpy as np
from collections import defaultdict

def create_data(dataframe_path):
    df = pd.read_csv(dataframe_path, sep =',')
    df = df.copy()
    df['annotation'] = df['annotation'].apply(lambda x: x.strip("[]'"))
    df['image_details'] = df['image_details'] + ' ' + df['annotation']
    df.to_csv(r'./data/indexed_data.csv', index=False, encoding='utf-8')     
    image_data = defaultdict(dict)
    for _, row in df.iterrows():
        image_id = row['image_id']
        image_data[image_id]['image_source'] = row['image_source']
        image_data[image_id]['image_details'] = row['image_details']
        image_data[image_id]['photographer'] = row['photographer']
        image_data[image_id]['image_name'] = row['image_name']
        
    with open(r'./data/image_data.pkl', 'wb') as f:
        pickle.dump(image_data, f)


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

if __name__ == '__main__':
    links_path = r'./config/links.json'
    with open(links_path) as f:
        data = json.load(f)
    print(data['links'])
    indexed_data = indexing.index_collection(data['links'])

    directory_path = r'./images'

    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    label = []
    image_name = []
    # file_path_short = file_paths[0:10]
    # counter = 0 
    # print(f'tThe len of file path is {len(file_path_short)}')
    for path in tqdm(file_paths):
        final_data = annotate.showresults(path)
        if not final_data[1]:
            image_name.append(path[9:].split('_', 1)[1])
            label.append(" ")
        else:
            label.append(final_data[1])
            image_name.append(path[9:].split('_', 1)[1])
            
    df_labels = pd.DataFrame(data = {'image_name': image_name, 
                                    'annotation': label})
    
    df_labels = df_labels.rename(columns={"image_name":"original_image_name"})
    
    df_merged = pd.merge(indexed_data, df_labels, on = "original_image_name")

    df_merged.to_csv(r'./data/indexed_data.csv', index=False, encoding='utf-8')   
    
    path = r'./data/indexed_data.csv'
    
    create_data(path)
    
    with open(r'./data/image_data.pkl', 'rb') as f:
        image_data = pickle.load(f)
    
    # documents= {image_id: image_data[image_id]['image_details'] for image_id in image_data}
    

    # inv_index = generate_index(documents)

    # with open(r'./data/inverted_index.pkl', 'wb') as f:
    #     pickle.dump(inv_index, f)
    

    
    
    
    
    
    
    
    
