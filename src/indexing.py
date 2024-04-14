from src import crawler
import pandas as pd 


def index_collection(urls):
    cols = ['photographer', 'image_source', 'url', 
               'original_image_name', 'image_details']
    data = pd.DataFrame(columns = cols)
    for current_url in urls:
        c = crawler.Crawler(current_url)
        data = data.append(c.scrape(), ignore_index=True)
    image_name = [f"image_{i}.jpg" for i in range(data.shape[0])] 
    data["image_name"] = image_name

    data.index.name = 'image_id'
    data.reset_index(inplace=True)
    data = data.sort_values(by='image_id')
    data['image_details']  = data['photographer'] + ' ' + data['image_details']
    return data

