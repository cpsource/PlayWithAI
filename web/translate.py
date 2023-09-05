# pip install googletrans==4.0.0-rc1

from googletrans import Translator

translator = Translator()
result = translator.translate('Bonjour', src='fr', dest='en')
print(result.text)  # Outputs: 'Hello'

