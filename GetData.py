import requests, csv, pulp
import numpy as np

output = open('gen1.csv', 'w+', newline ='')
writer = csv.writer(output)
writer.writerow(['number', 'name', 'total_stats', 'type1', 'type2'])

url = 'https://pokeapi.co/api/v2/pokemon/'

for number in range(810, 894):
    r = requests.get(url + str(number))
    name = r.json()['name']
    total_stats = sum([r.json()['stats'][i]['base_stat'] for i in range(5)])
    type1 = r.json()['types'][0]['type']['name']
    try:
        type2 = r.json()['types'][1]['type']['name']
        writer.writerow([number, name, total_stats, type1, type2])
    except:
        KeyError
        writer.writerow([number, name, total_stats, type1])
