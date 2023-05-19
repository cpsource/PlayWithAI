import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import re

def get_genus(wikipedia_url):
    html = urlopen(wikipedia_url) 
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', class_ = 'infobox biota')
    #print(table)
    foo = table.find(string=re.compile("<td>Genus:\n</td>\n.*</a>",re.MULTILINE))
    print(foo)

#    rows = table.find_all('tr')
#    for row in rows:
#        cells = row.find_all('td')
#        for cell in cells:
#            print("new cell")
#            print(cell)

if __name__ == '__main__':
    # Get the genus of the Wikipedia page for the dog
    dog_genus = get_genus('https://en.wikipedia.org/wiki/Dog')
    print(dog_genus)
    
