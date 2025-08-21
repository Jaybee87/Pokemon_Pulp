#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This uses GET API to generate a csv file that contain 8 columns:
# Pokemon Number, Pokemon Name, Total Stats, Ability1, Ability2, Ability3, Type1, Type2
# Original Creator: Geraint Palmer - palmergi1@cardiff.ac.uk
# Modified by: John Farrugia - jfarrugia87@gmail.com

import pandas, pulp
import numpy as np

# type chart values
type_IOAweaknesses = {
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
IOAtypeNames = []
IOAweaknesses = []
IOAstrengths = []
IOArotom = []

#Import data from spreadsheet
data = pandas.read_excel('IOApokemondata.xlsx', keep_default_na=False)
text_file = open("AdvancedTeams.txt", "w")

#Import all the data from csv
if IOAtypeNames == []:
    for index, row in data.iterrows():
        csvname = data.iat[0,1]
        csvtype1 = data.iat[0,2]
        csvtype2 = data.iat[0,3]  
        csvability1 = data.iat[0,4]
        csvability2 = data.iat [0,5]
        csvability3 = data.iat [0,6]
        
        weakness = list(np.array(type_IOAweaknesses[csvtype1]))
        strength = list(np.array(type_strenghts[csvtype1]))
        typename = csvtype1
        if csvtype2 !='':
            weakness = list(np.array(type_IOAweaknesses[csvtype1]) * np.array(type_IOAweaknesses[csvtype2]))
            strength = list(np.array(type_strenghts[csvtype1]) + np.array(type_strenghts[csvtype2]))
            typename = csvtype1 + " " + csvtype2
        
        if csvname != 'Rotom':
            if 'Rotom' in csvname:
                typename = typename + ' Rotom'
            if (csvability1 or csvability2 or csvability3) == 'Lightning Rod':
                pkmnname = typename + ' Lightning Rod'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Volt Absorb':
                pkmnname = typename + ' Volt Absorb'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Motor Drive':
                pkmnname = typename + ' Motor Drive'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Levitate':
                pkmnname = typename + ' Levitate'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Water Absorb':
                pkmnname = typename + ' Water Absorb'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Storm Drain':
                pkmnname = typename + ' Storm Drain'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Water Absorb':
                pkmnname = typename + ' Water Absorb'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Flash Fire':
                pkmnname = typename + ' Flash Fire'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Sap Sipper':
                pkmnname = typename + ' Sap Sipper'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Dry Skin':
                pkmnname = typename + ' Dry Skin'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Thick Fat':
                pkmnname = typename + ' Thick Fat'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Solid Rock':
                pkmnname = typename + ' Solid Rock'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            if (csvability1 or csvability2 or csvability3) == 'Filter':
                pkmnname = typename + ' Filter'
                if pkmnname not in IOAtypeNames:
                    IOAweaknesses.append(weakness)  
                    IOAtypeNames.append(pkmnname)
                    IOAstrengths.append(strength)
            elif typename not in IOAtypeNames:
                IOAweaknesses.append(weakness)  
                IOAtypeNames.append(typename)
                IOAstrengths.append(strength)
        
        data.drop(index, inplace=True)
    # Calculate Ability Bonuses
    for i in range(len(IOAtypeNames)):
        if 'Lightning Rod' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_electric'])
        elif 'Volt Absorb' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_electric'])
        elif 'Motor Drive' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_electric'])
        elif 'Levitate' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_ground'])
        elif 'Water Absorb' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_water'])
        elif 'Storm Drain' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_water'])
        elif 'Flash Fire' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_fire'])
        elif 'Sap Sipper' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['immune_grass'])
        elif 'Dry Skin' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['dryskin'])
        elif 'Thick Fat' in IOAtypeNames[i]:
            IOAweaknesses[i] = IOAweaknesses[i] * np.array(special['thickfat'])
        elif 'Solid Rock' in IOAtypeNames[i]:
            IOAweaknesses[i] = [r*0.75 if r > 1 else r*1 for r in IOAweaknesses[i]] # Only weakness values are reduced by 75%
        elif 'Filter' in IOAtypeNames[i]:
            IOAweaknesses[i] = [r*0.75 if r > 1 else r*1 for r in IOAweaknesses[i]] # Only weakness values are reduced by 75%
        

    for i in range(len(IOAtypeNames)):
        if 'Rotom' in IOAtypeNames[i]:
            IOArotom.append(1)
        else:
            IOArotom.append(0)
x = pulp.LpVariable.dicts("x", range(len(IOAweaknesses)), cat=pulp.LpBinary)

can_resist = [[1 if IOAweaknesses[i][t] < 1 else 0 for t in range(18)] for i in range(len(IOAweaknesses))]
is_weak = [[1 if IOAweaknesses[i][t] > 1 else 0 for t in range(18)] for i in range(len(IOAweaknesses))]
is_strong = [[1 if IOAstrengths[i][t] >= 1 else 0 for t in range(18)] for i in range(len(IOAweaknesses))]

def problem_solve():
    for i in range(1, 4):
        if goal == 1:
            prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMinimize)
        if i == 3:
            r = 2
        else:
            r = i
        text_file.write("With Minimum "+str(r)+" Resistances and Maximum "+str(i)+" IOAweaknesses"+'\n')
        prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMaximize)
        prob += objective_function
        prob += sum([x[t] for t in range(len(IOAweaknesses))]) == 6
        for tp in range(18):
            prob += sum([can_resist[i][tp] * x[i] for i in range(len(IOAweaknesses))]) >= r # This cannot be higher than 2
            prob += sum([is_weak[i][tp] * x[i] for i in range(len(IOAweaknesses))]) <= i
            prob += sum([is_strong[i][tp] * x[i] for i in range(len(IOAweaknesses))]) >= 1 # This cannot be higher than 1
            prob += sum([IOArotom[i] * x[i] for i in range(len(IOAweaknesses))]) <= 1 # This cannot be higher than 1
        prob.solve()
        for i in range(len(IOAweaknesses)):
            if x[i].value() > 0:
                text_file.write(str(IOAtypeNames[i])+'\n')
#goal = 1 
#resist = [sum([1 if w < 1 else 0 for w in IOAweaknesses[i]]) for i in range(len(IOAweaknesses))]
#objective_function = sum([resist[t] * x[t] for t in range(len(IOAweaknesses))])
#text_file.write("Max Resist"+'\n')
#problem_solve()
#text_file.write('\n')
#superresist = [sum([w if w < 0.5 else 0 for w in IOAweaknesses[i]]) for i in range(len(IOAweaknesses))]
#objective_function = sum([superresist[t] * x[t] for t in range(len(IOAweaknesses))])
#text_file.write("Max SuperResist"+'\n')
#problem_solve()
#text_file.write('\n')
goal = 0
#weak = [sum([1 if w > 1 else 0 for w in IOAweaknesses[i]]) for i in range(len(IOAweaknesses))]
#objective_function = sum([weak[t] * x[t] for t in range(len(IOAweaknesses))])
#text_file.write("Min Weak"+'\n')
#problem_solve()
#text_file.write('\n')
#superweak = [sum([1 if w > 2 else 0 for w in IOAweaknesses[i]]) for i in range(len(IOAweaknesses))]
#objective_function = sum([superweak[t] * x[t] for t in range(len(IOAweaknesses))])
#text_file.write("Min Superweak"+'\n')
#problem_solve()
text_file.write('\n')
text_file.write("Min Total"+'\n')
total = [sum(IOAweaknesses[i]) for i in range(len(IOAweaknesses))]
objective_function = sum([total[t] * x[t] for t in range(len(IOAweaknesses))])
problem_solve()
text_file.close()
