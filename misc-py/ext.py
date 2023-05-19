import re

line = "<td><a href=\"/wiki/Canis\" title=\"Canis\"><i>Canis</i></a>"

match = re.search(r"\>(.*?)\<a\>", line)

if match:
  print(match.group(1))

