import requests
from bs4 import BeautifulSoup

def get_genus(wikipedia_url):
    # Make a request to the Wikipedia URL
    response = requests.get(wikipedia_url)

    # Parse the HTML response
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the genus title
    #genus_title = soup.find('h1', class_='firstHeading').get_text()
    #genus_title = soup.find('dt', text='Genus').find_next_sibling('dd').text
    # Find the genus title
    # Find the scientific classification section
    scientific_classification_section = soup.find('table', class_='infobox biota')
    print(scientific_classification_section)
    soup1= BeautifulSoup(scientific_classification_section,'html.parser');

    # Find the genus title
    genus_title = soup1.find('td', text='Genus:')
    print(genus_title)

    # Extract the genus from the title
    genus = genus_title.split(' ')[0]

    return genus

if __name__ == '__main__':
    # Get the genus of the Wikipedia page for the dog
    dog_genus = get_genus('https://en.wikipedia.org/wiki/Dog')
    print(dog_genus)

