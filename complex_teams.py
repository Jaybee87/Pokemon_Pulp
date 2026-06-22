from flask_app import app
from flask import Flask, redirect, render_template, request, url_for, session
import pulp, pandas
import numpy as np
from type_chart import *

# Used for Complex Team-Calculator

IOApkmnname = []
IOAweaknesses = []
IOAstrengths = []
IOArotom = []
#moves
IOAfakeout = []
IOAfollowme = []
IOAragepowder = []
IOAnuzzle = []
IOAtailwind = []
IOAtrickroom = []
#abilities
IOAintimidate = []
IOAdefiant = []
IOAcompetitive = []
IOAprankster = []
IOAsand = []
IOArain = []
IOAsnow = []
IOAsun = []
IOAchlorophyll = []
IOAswiftswim = []
IOAsandrush = []
IOAslushrush = []
#Stats
IOAhp = []
IOAattack = []
IOAdef = []
IOAspattack = []
IOAspdef = []
IOAspeeds = []
IOAtotalstats = []
IOAdefpower = []
IOAoffpower = []
IOAoffspe = []

TCTpkmnname = []
TCTweaknesses = []
TCTstrengths = []
TCTrotom = []
#moves
TCTfakeout = []
TCTfollowme = []
TCTragepowder = []
TCTnuzzle = []
TCTtailwind = []
TCTtrickroom = []
#abilities
TCTintimidate = []
TCTdefiant = []
TCTcompetitive = []
TCTprankster = []
TCTsand = []
TCTrain = []
TCTsnow = []
TCTsun = []
TCTchlorophyll = []
TCTswiftswim = []
TCTsandrush = []
TCTslushrush = []
#Stats
TCThp = []
TCTattack = []
TCTdef = []
TCTspattack = []
TCTspdef = []
TCTspeeds = []
TCTtotalstats = []
TCTdefpower = []
TCToffpower = []
TCToffspe = []

@app.route('/teambuilder_complex/', methods=['POST'])
def complex_teams():
    session["outputs"].clear()
    session.modified = True
    teamtype = request.form["teamtype"]
    available = request.form["available"]
    weatherteam = request.form["weatherteam"]
    weatherspeed = request.form["weatherspeed"]
    if teamtype == "None":
        session["outputs"].append("You have to select a teamtype from the dropdown list")
    else:
        session["outputs"].append(teamtype)
        if available == "IOA":
            pkmnnames = IOApkmnname
            weaknesses = IOAweaknesses
            strengths = IOAstrengths
            rotoms = IOArotom
            #moves
            fakeout = IOAfakeout
            followme = IOAfollowme
            ragepowder = IOAragepowder
            nuzzle = IOAnuzzle
            tailwind = IOAtailwind
            trickroom = IOAtrickroom
            #abilities
            intimidate = IOAintimidate
            defiant = IOAdefiant
            competitive = IOAcompetitive
            prankster = IOAprankster
            sandstream = IOAsand
            drizzle = IOArain
            drought = IOAsun
            snowwarning = IOAsnow
            chlorophyll = IOAchlorophyll
            swiftswim = IOAswiftswim
            sandrush = IOAsandrush
            slushrush = IOAslushrush
            #stats
            hp = IOAhp
            attack = IOAattack
            defe = IOAdef
            spattack = IOAspattack
            spdef = IOAspdef
            speeds = IOAspeeds
            totalstats = IOAtotalstats
            defpower = IOAdefpower
            offpower = IOAoffpower
            offspe = IOAoffspe
            data = pandas.read_excel('/home/JaybeeVGC/mysite/IOApokemondata.xlsx', keep_default_na=False)
        elif available == "TCT":
            pkmnnames = TCTpkmnname
            weaknesses = TCTweaknesses
            strengths = TCTstrengths
            rotoms = TCTrotom
            #moves
            fakeout = TCTfakeout
            followme = TCTfollowme
            ragepowder = TCTragepowder
            nuzzle = TCTnuzzle
            tailwind = TCTtailwind
            trickroom = TCTtrickroom
            #abilities
            intimidate = TCTintimidate
            defiant = TCTdefiant
            competitive = TCTcompetitive
            prankster = TCTprankster
            sandstream = TCTsand
            drizzle = TCTrain
            drought = TCTsun
            snowwarning = TCTsnow
            chlorophyll = TCTchlorophyll
            swiftswim = TCTswiftswim
            sandrush = TCTsandrush
            slushrush = TCTslushrush
            #stats
            hp = TCThp
            attack = TCTattack
            defe = TCTdef
            spattack = TCTspattack
            spdef = TCTspdef
            speeds = TCTspeeds
            totalstats = TCTtotalstats
            defpower = TCTdefpower
            offpower = TCToffpower
            offspe = TCToffspe
            data = pandas.read_excel('/home/JaybeeVGC/mysite/TCTpokemondata.xlsx', keep_default_na=False)
        if pkmnnames == []:
            for index, row in data.iterrows():
                csvcondition = data.iat[0,0]
                csvname = data.iat[0,1]
                csvtype1 = data.iat[0,2]
                csvtype2 = data.iat[0,3]
                csvability1 = data.iat[0,4]
                csvability2 = data.iat [0,5]
                csvability3 = data.iat [0,6]
                csvhp = data.iat [0,7]
                csvattack = data.iat [0,8]
                csvdef = data.iat [0,9]
                csvspattack = data.iat [0,10]
                csvspdef = data.iat [0,11]
                csvspeed = data.iat [0,12]
                csvtotalstats = data.iat [0,13]
                csvdefpower = data.iat [0,14]
                csvoffpower = data.iat [0,15]
                csvoffspe = data.iat [0,16]
                csvevolve = data.iat [0,17]

                def add_to_list():
                    weakness = list(np.array(type_weaknesses[csvtype1]))
                    strength = list(np.array(type_strengths[csvtype1]))
                    if csvtype2 !='':
                        weakness = list(np.array(type_weaknesses[csvtype1]) * np.array(type_weaknesses[csvtype2]))
                        strength = list(np.array(type_strengths[csvtype1]) + np.array(type_strengths[csvtype2]))
                    if ability == 'Lightning Rod':
                        weakness *= np.array(special['immune_electric'])
                    elif ability == 'Volt Absorb':
                        weakness *= np.array(special['immune_electric'])
                    elif ability == 'Motor Drive':
                        weakness *= np.array(special['immune_electric'])
                    elif ability == 'Levitate':
                        weakness *= np.array(special['immune_ground'])
                    elif ability == 'Water Absorb':
                        weakness *= np.array(special['immune_water'])
                    elif ability == 'Storm Drain':
                        weakness *= np.array(special['immune_water'])
                    elif ability == 'Flash Fire':
                        weakness *= np.array(special['immune_fire'])
                    elif ability == 'Sap Sipper':
                        weakness *= np.array(special['immune_grass'])
                    elif ability == 'Dry Skin':
                        weakness *= np.array(special['dryskin'])
                    elif ability == 'Thick Fat':
                        weakness *= np.array(special['thickfat'])
                    elif ability == 'Solid Rock':
                        weakness = [r*0.75 if r > 1 else r*1 for r in weakness] # Only weakness values are reduced by 75%
                    elif ability == 'Filter':
                        weakness = [r*0.75 if r > 1 else r*1 for r in weakness] # Only weakness values are reduced by 75%
                    pkmnnames.append(pkmnname)
                    weaknesses.append(weakness)
                    strengths.append(strength)
                    #stats
                    hp.append(csvhp)
                    attack.append(csvattack)
                    defe.append(csvdef)
                    spattack.append(csvspattack)
                    spdef.append(csvspdef)
                    speeds.append(csvspeed)
                    totalstats.append(csvtotalstats)
                    defpower.append(csvdefpower)
                    offpower.append(csvoffpower)
                    offspe.append(csvoffspe)

                if csvevolve == 'N' and "Legendary" not in csvcondition:
                    if 'Fakeout' in csvcondition:
                        csvname = csvname + ' Fakeout'
                    if 'RagePowder' in csvcondition:
                        csvname = csvname + ' RagePowder'
                    if 'FollowMe' in csvcondition:
                        csvname = csvname + ' FollowMe'
                    if 'Tailwind' in csvcondition:
                        csvname = csvname + ' Tailwind'
                    if 'Nuzzle' in csvcondition:
                        csvname = csvname + ' Nuzzle'
                    if 'TR' in csvcondition:
                        csvname = csvname + ' Trick Room'

                    if csvability1 != '':
                        pkmnname = csvname + ' ' + csvability1
                        ability = csvability1
                        add_to_list()
                    if csvability2 != '':
                        pkmnname = csvname + ' ' + csvability2
                        ability = csvability2
                        add_to_list()
                    if csvability3 != '':
                        pkmnname = csvname + ' ' + csvability3
                        ability = csvability3
                        add_to_list()

                data.drop(index, inplace=True)

            # Calculate Ability Bonuses
            for i in range(len(pkmnnames)):
                if 'Rotom' in pkmnnames[i]:
                    rotoms.append(1)
                else:
                    rotoms.append(0)
                # Moves
                if 'Fakeout' in pkmnnames[i]:
                    fakeout.append(1)
                else:
                    fakeout.append(0)
                if 'Tailwind' in pkmnnames[i]:
                    tailwind.append(1)
                else:
                    tailwind.append(0)
                if 'Nuzzle' in pkmnnames[i]:
                    nuzzle.append(1)
                else:
                    nuzzle.append(0)
                if 'FollowMe' in pkmnnames[i]:
                    followme.append(1)
                else:
                    followme.append(0)
                if 'RagePowder' in pkmnnames[i]:
                    ragepowder.append(1)
                else:
                    ragepowder.append(0)
                if 'Intimidate' in pkmnnames[i]:
                    intimidate.append(1)
                else:
                    intimidate.append(0)
                # Abilities
                if 'Defiant' in pkmnnames[i]:
                    defiant.append(1)
                else:
                    defiant.append(0)
                if 'Competitive' in pkmnnames[i]:
                    competitive.append(1)
                else:
                    competitive.append(0)
                if 'Prankster' in pkmnnames[i]:
                    prankster.append(1)
                else:
                    prankster.append(0)
                if 'Drought' in pkmnnames[i]:
                    drought.append(1)
                else:
                    drought.append(0)
                if 'Snow Warning' in pkmnnames[i]:
                    snowwarning.append(1)
                else:
                    snowwarning.append(0)
                if 'Sand Stream' in pkmnnames[i]:
                    sandstream.append(1)
                else:
                    sandstream.append(0)
                if 'Drizzle' in pkmnnames[i]:
                    drizzle.append(1)
                else:
                    drizzle.append(0)
                if 'Trick Room' in pkmnnames[i]:
                    trickroom.append(1)
                else:
                    trickroom.append(0)

                if 'Chlorophyll' in pkmnnames[i]:
                    chlorophyll.append(1)
                else:
                    chlorophyll.append(0)
                if 'Swift Swim' in pkmnnames[i]:
                    swiftswim.append(1)
                else:
                    swiftswim.append(0)
                if 'Sand Rush' in pkmnnames[i]:
                    sandrush.append(1)
                else:
                    sandrush.append(0)
                if 'Slush Rush' in pkmnnames[i]:
                    slushrush.append(1)
                else:
                    slushrush.append(0)


        specify1 = request.form["specify1"]     # Input Variable
        if specify1 != "":
            session["specifics1"] = []              # List specific to users specific selection
            for i in range(len(pkmnnames)):
                if pkmnnames[i].startswith(specify1) and not pkmnnames[i].startswith(specify1+'-'): # Starts with name, but filters out regional variant possibilities
                    session["specifics1"].append(1)
                else:
                    session["specifics1"].append(0)
            if 1 not in session["specifics1"]:  # Check to see it found what it was looking for
                session["outputs"].append(specify1+" Not Available")
                specify1 = "None"
        else:
            specify1 = "None"

        x = pulp.LpVariable.dicts("x", range(len(weaknesses)), cat=pulp.LpBinary)

        can_resist = [[1 if weaknesses[i][t] < 1 else 0 for t in range(18)] for i in range(len(weaknesses))]
        is_weak = [[1 if weaknesses[i][t] > 1 else 0 for t in range(18)] for i in range(len(weaknesses))]
        is_strong = [[1 if strengths[i][t] >= 1 else 0 for t in range(18)] for i in range(len(weaknesses))]

        def problem_solve():
            for i in range(1, 4):
                if teamtype == "MinSpeed":
                    prob = pulp.LpProblem("PerfectPokemonTeam", pulp.LpMinimize)
                else:
                    prob = pulp.LpProblem("PerfectPokemonTeam", pulp.LpMaximize)
                if i == 3:
                    r = 2
                else:
                    r = i
                session["outputs"].append("With Minimum "+str(r)+" Resistances and Maximum "+str(i)+" Weaknesses")
                prob += objective_function
                prob += sum([x[t] for t in range(len(weaknesses))]) == 6
                prob += sum([rotoms[i] * x[i] for i in range(len(weaknesses))]) <= 1 # Ensure no more than 1x Rotom on a team
                if specify1 != "None":
                    prob += sum([session["specifics1"][i] * x[i] for i in range(len(weaknesses))]) == 1
                try:
                    if request.form["check1"] == "fakeout":
                        prob += sum([fakeout[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with move Fakeout
                except:
                    KeyError
                try:
                    if request.form["check2"] == "intimidate":
                        prob += sum([intimidate[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with ability Intimidate
                except:
                    KeyError
                try:
                    if request.form["check3"] == "defiant":
                        prob += sum([defiant[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with ability Defiant
                except:
                    KeyError
                try:
                    if request.form["check4"] == "competitive":
                        prob += sum([competitive[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with ability Competitive
                except:
                    KeyError
                try:
                    if request.form["check5"] == "prankster":
                        prob += sum([prankster[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with ability Prankster
                except:
                    KeyError
                try:
                    if request.form["check6"] == "followme":
                        prob += sum([followme[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with move FollowMe
                except:
                    KeyError
                try:
                    if request.form["check7"] == "ragepowder":
                        prob += sum([ragepowder[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with move RagePowder
                except:
                    KeyError
                try:
                    if request.form["check8"] == "tailwind":
                        prob += sum([tailwind[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with move Tailwind
                except:
                    KeyError
                try:
                    if request.form["check9"] == "nuzzle":
                        prob += sum([nuzzle[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with move Nuzzle
                except:
                    KeyError

                if weatherteam == "Rain":
                    prob += sum([drizzle[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability Drizzle
                elif weatherteam == "Sunshine":
                    prob += sum([drought[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability Drought
                elif weatherteam == "Sandstorm":
                    prob += sum([sandstream[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability Sandstream
                elif weatherteam == "Hail":
                    prob += sum([snowwarning[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability Snow Warning

                if weatherspeed == "chlorophyll":
                    prob += sum([chlorophyll[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability chlorophyll
                elif weatherspeed == "swiftswim":
                    prob += sum([swiftswim[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability swiftswim
                elif weatherspeed == "sandrush":
                    prob += sum([sandrush[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability sandrush
                elif weatherspeed == "slushrush":
                    prob += sum([slushrush[i] * x[i] for i in range(len(weaknesses))]) >= 1 # Atleast one with Ability slushrush


                for tp in range(18):
                    prob += sum([can_resist[i][tp] * x[i] for i in range(len(weaknesses))]) >= r # This cannot be higher than 2
                    prob += sum([is_weak[i][tp] * x[i] for i in range(len(weaknesses))]) <= i
                    prob += sum([is_strong[i][tp] * x[i] for i in range(len(weaknesses))]) >= 1 # This cannot be higher than 1
                prob.solve()
                # Check to see if the Problem is Feasible
                count = 0
                for i in range(len(weaknesses)):
                    if x[i].value() > 0:
                        count += 1
                if count == 6:
                    for i in range(len(weaknesses)):
                        if x[i].value() > 0:
                            session["outputs"].append(str(pkmnnames[i]))
                else:
                    session["outputs"].append("Problem is infeasible")

        if teamtype == "MaxHP":
            objective_function = sum([hp[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxAttack":
            objective_function = sum([attack[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxDefense":
            objective_function = sum([defe[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxSpecialAttack":
            objective_function = sum([spattack[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxSpecialDefense":
            objective_function = sum([spdef[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxSpeed":
            objective_function = sum([speeds[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MaxTotalStats":
            objective_function = sum([totalstats[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "DefensivePower":
            objective_function = sum([defpower[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "OffensivePower":
            objective_function = sum([offpower[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "OffensiveSpeed":
            objective_function = sum([offspe[t] * x[t] for t in range(len(weaknesses))])
        elif teamtype == "MinSpeed":
            objective_function = sum([speeds[t] * x[t] for t in range(len(weaknesses))])
        problem_solve()

    return redirect(url_for('classic'))

