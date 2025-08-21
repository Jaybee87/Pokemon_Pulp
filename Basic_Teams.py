import pulp, itertools
import numpy as np
from type_chart import *

order = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
text_file = open("BasicTeams.txt", "w")
# https://bulbapedia.bulbagarden.net/wiki/List_of_type_combinations_by_abundance#All_type_combinations_and_their_abundance
unused = {
    ('normal', 'ice'),      # Not yet assigned
    ('normal', 'poison'),   # Not yet assigned
    ('normal', 'bug'),      # Not yet assigned
    ('normal', 'rock'),     # Not yet assigned
    ('normal', 'ghost'),    # Not yet assigned
    ('normal', 'steel'),    # Not yet assigned
    ('fire', 'grass'),      # Not yet assigned
    ('electric', 'fighting'),   # Not yet assigned
    ('ice', 'poison'),      # Not yet assigned
    ('fighting', 'ground'), # Not yet assigned
    ('fighting', 'fairy'),  # Not yet assigned
    ('poison', 'steel'),    # Not yet assigned
    ('ground', 'fairy'),    # Not yet assigned
    ('bug', 'dragon'),      # Not yet assigned
    ('bug', 'dark'),        # Not yet assigned
    ('rock', 'ghost'),      # Not yet assigned
    ('fire', 'fairy'),      # Not yet assigned
}
IOA = {
    ('normal', 'fire'),     # Littleo, Pyroar
    ('normal', 'water'),    # Bibarel
    ('normal', 'grass'),    # Sawsbuck, Deerling
    ('fire', 'ground'),     # Numel/Camerupt/Groudon
    ('fire', 'steel'),      # Heatran
    ('ground', 'bug'),      # Nincada
    ('water', 'steel'),     # Empoleon
    ('electric', 'ghost'),  # Rotom-Normal
    ('fighting', 'ghost'),  # Marshadow
    ('poison', 'ground'),   # NidoKing/Queen
    ('ground', 'flying'),   # Gligar, Gliscor, Landorous
    ('poison', 'flying'),   # Zubat, Golbat, Crobat
    ('psychic', 'ghost'),   # Lunala/Necrozma
    ('dragon', 'fairy'),    # Mega Altaria
    ('rock', 'steel'),      # Aron, Lairon, Aggron, Shieldon, Bastiodon, Probopass, Stakataka
    }
TCT = {
    ('normal', 'fire'),     # Littleo, Pyroar
    ('normal', 'water'),    # Bibarel
    ('normal', 'grass'),    # Sawsbuck, Deerling
    ('fire', 'ground'),     # Numel/Camerupt/Groudon
    ('ground', 'bug'),      # Nincada
    ('water', 'steel'),     # Empoleon
    ('electric', 'ghost'),  # Rotom-Normal
    ('fighting', 'ghost'),  # Marshadow
    ('psychic', 'ghost'),   # Lunala/Necrozma
    ('dragon', 'fairy'),    # Mega Altaria
    }

used = {
    ('fire', 'bug'),
    ('fire', 'rock'),
    ('fire', 'steel'),
    ('water', 'grass'),
    ('electric', 'ground'),
    ('electric', 'flying'),
    ('grass', 'poison'),
    ('grass', 'ground'),
    ('grass', 'flying'),
    ('grass', 'bug'),
    ('fighting', 'flying'),
    ('fighting', 'psychic'),
    ('fighting', 'rock'),
    ('fighting', 'dark'),
    ('poison', 'ground'),
    ('poison', 'fairy'),
    ('flying', 'bug'),
    ('flying', 'rock'),
    ('psychic', 'dark'),
    ('rock', 'steel'),
    ('ghost', 'dark'),
    ('dark', 'fairy'),
    ('steel', 'fairy'),
}

available = 'TCT'
if available == "IOA":
    unused.update(IOA)
elif available == "TCT":
    unused.update(TCT)
pair_order = order[:]
for pair in itertools.combinations(order, 2): #Generate all possible dual-type combinations
    #if pair not in unused:
    if pair in used:
        print(pair)
        #Calculate Weaknesses
        pair_vector = np.array(type_weaknesses[pair[0]]) * np.array(type_weaknesses[pair[1]])
        type_weaknesses[pair] = list(pair_vector)
        #Calculate Strengths
        pair_vector = np.array(type_strengths[pair[0]]) + np.array(type_strengths[pair[1]])
        type_strengths[pair] = list(pair_vector)
        #Add to list
        pair_order.append(pair)
    #else:
    #    outputs.append(pair)

can_resist = [[1 if type_weaknesses[pair][t] < 1 else 0 for t in range(len(order))] for pair in pair_order]
is_weak = [[1 if type_weaknesses[pair][t] > 1 else 0 for t in range(len(order))] for pair in pair_order]
is_strong = [[1 if type_strengths[pair][t] >= 1 else 0 for t in range(len(order))] for pair in pair_order]

x = pulp.LpVariable.dicts("x", range(len(pair_order)), cat=pulp.LpBinary)

def problem_solve():
    for i in range(1,4):
        if goal == 1:
            prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMinimize)
        else:
            prob = pulp.LpProblem("PokemonpairCombos", pulp.LpMaximize)
        prob += objective_function
        checksize = 6
        prob += sum([x[t] for t in range(len(pair_order))]) == checksize
        if i <= 2:
            r = i
        else:
            r = 2
        text_file.write("With Minimum "+str(r)+" Resistances and Maximum "+str(i)+" Weaknesses"+'\n')
        for tp in range(len(order)):
            prob += sum([can_resist[pair][tp] * x[pair] for pair in range(len(pair_order))]) >= r # This cannot be higher than 2
            prob += sum([is_weak[pair][tp] * x[pair] for pair in range(len(pair_order))]) <= i
            prob += sum([is_strong[pair][tp] * x[pair] for pair in range(len(pair_order))]) >= 1 # This cannot be higher than 1

        prob.solve()
        # Check to see if the Problem is Feasible
        count = 0
        for i in range(len(pair_order)):
            if x[i].value() > 0:
                count += 1
        if count == checksize:
            for i in range(len(pair_order)):
                if x[i].value() > 0:
                    text_file.write(str(pair_order[i])+'\n')
        else:
            text_file.write("Problem is infeasible"+'\n')

goal = 1
text_file.write('\n')
text_file.write("Max Resist"+'\n')
resist = [sum([1 if w < 1 else 0 for w in type_weaknesses[pair]]) for pair in pair_order]
objective_function = sum([resist[t] * x[t] for t in range(len(pair_order))])
problem_solve()
text_file.write('\n')
text_file.write("Max SuperResist"+'\n')
superresist = [sum([1 if w < 0.5 else 0 for w in type_weaknesses[pair]]) for pair in pair_order]
objective_function = sum([superresist[t] * x[t] for t in range(len(pair_order))])
problem_solve()
goal = 0
text_file.write('\n')
text_file.write("Min Weak"+'\n')
weak = [sum([1 if w > 1 else 0 for w in type_weaknesses[pair]]) for pair in pair_order]
objective_function = sum([weak[t] * x[t] for t in range(len(pair_order))])
problem_solve()
text_file.write('\n')
text_file.write("Min Superweak"+'\n')
superweak = [sum([1 if w > 2 else 0 for w in type_weaknesses[pair]]) for pair in pair_order]
objective_function = sum([superweak[t] * x[t] for t in range(len(pair_order))])
problem_solve()
text_file.write('\n')
text_file.write("Min Total"+'\n')
total = [sum(type_weaknesses[pair]) for pair in pair_order]
objective_function = sum([total[t] * x[t] for t in range(len(pair_order))])
problem_solve()
text_file.write('\n')
text_file.close()
