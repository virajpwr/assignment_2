import mechanicalsoup
import requests
import pandas as pd
import os
import re

class Crawler:
    def __init__(self, url):
        self.url = url
        self.browser = mechanicalsoup.StatefulBrowser()
        self.browser.open(url)
        self.image_source = []
        self.image_link = []
        self.urls = []
        self.contents = []
        self.folder_path = r'./images/'
        
    def get_references(self):
        self.photographer = [img.get("alt").replace('photo copyright ', '') for img in self.browser.page.select('div.content_div_thumb img')]
        self.image_source = [img.get("src") for img in self.browser.page.select('div.content_div_thumb img')]
        self.image_link = [a.get("href") for a in self.browser.page.select('div.content_div_thumb a')]
        self.image_link = [item for item in self.image_link if not item.startswith('http')]
        url_2 = "https://openphoto.net"
        self.urls = [url_2 + url for url in self.image_link]
        
    def get_contents(self, url):
        self.browser.open(url)
        content = [content.get('content') for content in self.browser.page.select('meta')]
        contents = content[2:4]
        return contents
    
    def download_images(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            
        for _, url in enumerate(self.image_source):
            response = requests.get(url)
            img_name = url.split('/')[-1]
            if response.status_code == 200:
                file_path = os.path.join(self.folder_path, f"image_{img_name}")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Failed to download image from {url}")
    
    def scrape(self):
        self.get_references()
        self.contents = [self.get_contents(self.urls[i]) for i in range(len(self.urls))]
        image_details = [self.contents[i][0] + self.contents[i][1] for i in range(len(self.contents))]
        orginal_image_name = [self.image_source[i].split('/')[-1] for i in range(len(self.image_source))] 
        indexed_images = pd.DataFrame({'photographer': self.photographer, 'image_source': self.image_source, 
                                       'url': self.urls,'original_image_name': orginal_image_name,
                                       'image_details': image_details})
        self.download_images()
        return indexed_images
