#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Original Creator: Geraint Palmer - palmergi1@cardiff.ac.uk
# Modified by: John Farrugia - jfarrugia87@gmail.com

import pandas, pulp
import numpy as np

# type chart values
type_weaknesses = {
    'normal': [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    'fire': [1, 0.5, 2, 1, 0.5, 0.5, 1, 1, 2, 1, 1, 0.5, 2, 1, 1, 1, 0.5, 0.5],
    'water': [1, 0.5, 0.5, 2, 2, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1],
    'electric': [1, 1, 1, 0.5, 1, 1, 1, 1, 2, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 1],
    'grass': [1, 2, 0.5, 0.5, 0.5, 2, 1, 2, 0.5, 2, 1, 2, 1, 1, 1, 1, 1, 1],
    'ice': [1, 2, 1, 1, 1, 0.5, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1],
    'fighting': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0.5, 0.5, 1, 1, 0.5, 1, 2],
    'poison': [1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 2, 1, 2, 0.5, 1, 1, 1, 1, 1, 0.5],
    'ground': [1, 1, 2, 0, 2, 2, 1, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 1, 1],
    'flying': [1, 1, 1, 2, 0.5, 2, 0.5, 1, 0, 1, 1, 0.5, 2, 1, 1, 1, 1, 1],
    'psychic': [1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 2, 1, 2, 1, 2, 1, 1],
    'bug': [1, 2, 1, 1, 0.5, 1, 0.5, 1, 0.5, 2, 1, 1, 2, 1, 1, 1, 1, 1],
    'rock': [0.5, 0.5, 2, 1, 2, 1, 2, 0.5, 2, 0.5, 1, 1, 1, 1, 1, 1, 2, 1],
    'ghost': [0, 1, 1, 1, 1, 1, 0, 0.5, 1, 1, 1, 0.5, 1, 2, 1, 2, 1, 1],
    'dragon': [1, 0.5, 0.5, 0.5, 0.5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2],
    'dark': [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 2, 1, 0.5, 1, 0.5, 1, 2],
    'steel': [0.5, 2, 1, 1, 0.5, 0.5, 2, 0, 2, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5],
    'fairy': [1, 1, 1, 1, 1, 1, 0.5, 2, 1, 1, 1, 0.5, 1, 1, 0, 0.5, 2, 1]
}

# type chart values, supereffective = 1 else 0
type_strenghts = {
    'normal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'fire': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'water': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'electric': [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'grass': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'ice': [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'fighting': [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    'poison': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'ground': [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'flying': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'psychic': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'bug': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'rock': [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    'ghost': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'dragon': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'dark': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'steel': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    'fairy': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
}

# Special modifiers
special = {
    'dryskin': [1, 1.25, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'thickfat': [1, .5, 1, 1, 1, .5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'immune_ground': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'immune_electric': [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'immune_water': [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'immune_fire': [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'immune_grass': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    
    
# Store Combinations and Weakness Values
typenames = []
weaknesses = []
strengths = []

#Import data from spreadsheet
data = pandas.read_excel('combinations.xlsx', keep_default_na=False)
text_file = open("AdvancedTeams.txt", "w")

#Import all the data from csv
for index, row in data.iterrows():
    csvtype1 = data.iat[0,0]
    csvtype2 = data.iat[0,1]  
    csvability1 = data.iat[0,2]
    csvability2 = data.iat [0,3]
    
    weakness = list(np.array(type_weaknesses[csvtype1]))
    strength = list(np.array(type_strenghts[csvtype1]))
    typename = csvtype1
    if csvtype2 !='':
        weakness = list(np.array(type_weaknesses[csvtype1]) * np.array(type_weaknesses[csvtype2]))
        strength = list(np.array(type_strenghts[csvtype1]) + np.array(type_strenghts[csvtype2]))
        typename = csvtype1 + " " + csvtype2
    
    if csvability1 !='':
        pkmnname = typename + " " + csvability1
        weaknesses.append(weakness)  
        typenames.append(pkmnname)
        strengths.append(strength)
    else:
        weaknesses.append(weakness)  
        typenames.append(typename)
        strengths.append(strength)
    
    if csvability2 !='':
        pkmnname = typename + " " + csvability2
        weaknesses.append(weakness)  
        typenames.append(pkmnname)
        strengths.append(strength)
    else:
        weaknesses.append(weakness)  
        typenames.append(typename)
        strengths.append(strength)
    
    data.drop(index, inplace=True)

# Calculate Ability Bonuses
for i in range(len(typenames)):
    if 'Lightning Rod' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_electric'])
    elif 'Volt Absorb' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_electric'])
    elif 'Motor Drive' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_electric'])
    elif 'Levitate' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_ground'])
    elif 'Water Absorb' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_water'])
    elif 'Storm Drain' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_water'])
    elif 'Flash Fire' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_fire'])
    elif 'Sap Sipper' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['immune_grass'])
    elif 'Dry Skin' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['dryskin'])
    elif 'Thick Fat' in typenames[i]:
        weaknesses[i] = weaknesses[i] * np.array(special['thickfat'])
    elif 'Solid Rock' in typenames[i]:
        weaknesses[i] = [r*0.75 if r > 1 else r*1 for r in weaknesses[i]] # Only weakness values are reduced by 75%
    elif 'Filter' in typenames[i]:
        weaknesses[i] = [r*0.75 if r > 1 else r*1 for r in weaknesses[i]] # Only weakness values are reduced by 75%

x = pulp.LpVariable.dicts("x", range(len(weaknesses)), cat=pulp.LpBinary)

can_resist = [[1 if weaknesses[i][t] < 1 else 0 for t in range(18)] for i in range(len(weaknesses))]
is_weak = [[1 if weaknesses[i][t] > 1 else 0 for t in range(18)] for i in range(len(weaknesses))]
is_strong = [[1 if strengths[i][t] >= 1 else 0 for t in range(18)] for i in range(len(weaknesses))]

def minimise_solve():
    for i in range(3):
        i=i+1
        if i == 3:
            r = 2
        else:
            r = i
        text_file.write("With Minimum "+str(r)+" Resistances and Maximum "+str(i)+" Weaknesses"+'\n')
        prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMinimize)
        prob += objective_function
        prob += sum([x[t] for t in range(len(weaknesses))]) == 6
        for tp in range(18):
            prob += sum([can_resist[i][tp] * x[i] for i in range(len(weaknesses))]) >= r # This cannot be higher than 2
            prob += sum([is_weak[i][tp] * x[i] for i in range(len(weaknesses))]) <= i
            prob += sum([is_strong[i][tp] * x[i] for i in range(len(weaknesses))]) >= 1 # This cannot be higher than 1
        prob.solve()
        for i in range(len(weaknesses)):
            if x[i].value() > 0:
                text_file.write(str(typenames[i])+'\n')
                    

def maximise_solve():
    for i in range(3):
        i=i+1
        if i == 3:
            r = 2
        else:
            r = i
        text_file.write("With Minimum "+str(r)+" Resistances and Maximum "+str(i)+" Weaknesses"+'\n')
        prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMaximize)
        prob += objective_function
        prob += sum([x[t] for t in range(len(weaknesses))]) == 6
        for tp in range(18):
            prob += sum([can_resist[i][tp] * x[i] for i in range(len(weaknesses))]) >= r # This cannot be higher than 2
            prob += sum([is_weak[i][tp] * x[i] for i in range(len(weaknesses))]) <= i
            prob += sum([is_strong[i][tp] * x[i] for i in range(len(weaknesses))]) >= 1 # This cannot be higher than 1
        prob.solve()
        for i in range(len(weaknesses)):
            if x[i].value() > 0:
                text_file.write(str(typenames[i])+'\n')

resist = [sum([1 if w < 1 else 0 for w in weaknesses[i]]) for i in range(len(weaknesses))]
objective_function = sum([resist[t] * x[t] for t in range(len(weaknesses))])
text_file.write("Max Resist"+'\n')
maximise_solve()
text_file.write('\n')
superresist = [sum([w if w < 0.5 else 0 for w in weaknesses[i]]) for i in range(len(weaknesses))]
objective_function = sum([superresist[t] * x[t] for t in range(len(weaknesses))])
text_file.write("Max SuperResist"+'\n')
maximise_solve()
text_file.write('\n')
weak = [sum([1 if w > 1 else 0 for w in weaknesses[i]]) for i in range(len(weaknesses))]
objective_function = sum([weak[t] * x[t] for t in range(len(weaknesses))])
text_file.write("Min Weak"+'\n')
minimise_solve()
text_file.write('\n')
superweak = [sum([1 if w > 2 else 0 for w in weaknesses[i]]) for i in range(len(weaknesses))]
objective_function = sum([superweak[t] * x[t] for t in range(len(weaknesses))])
text_file.write("Min Superweak"+'\n')
minimise_solve()
text_file.write('\n')
text_file.close()
