import pandas as pd                 
import numpy as np                   

missing = np.nan 
series_obj = pd.Series(['row 1', 'row2', missing, 'row 4', 'row5', 'row 6', missing, 'row 8' ])

series_obj.isnull()

np.random.seed(25) 
DF_obj = pd.DataFrame(np.random.randn(36).reshape(6,6))

DF_obj.ix[3:5,0] = missing
DF_obj.ix[1:4,5] = missing

DF_obj.fillna(0)
filled_DF = DF_obj.fillna({0: 0.1, 5:1.25})
fill_DF = DF_obj.fillna(method='ffill')

DF_obj.isnull().sum() # counts the null values of the line
DF_no_NaN = DF_obj.dropna()

DF_obj.drop_duplicates(['column 3']) # If duplicate values are found in this column, the entire row is dropped

concat = pd.concat([fill_DF, filled_DF], axis = 1)
concat2 = pd.concat([fill_DF, filled_DF], axis = 0)

droppo = concat.drop([0,2]) #axis = 1 to drop columns



new_df = pd.DataFrame.join(DF_obj, series_obj) #join or append to merge datasets


#OBJECT ORIENTED PLOTTING

# Step 1: Create a blank figure object
# Step 2: Add axes to the figure
# Step 3: Generate plot(s) within the figure object
# Step 4: Specify plotting and layout parameters for the plots within the figure
#-> also important: generate subplots


# PYTHON DESIGN PATTERN

cities, years = [],[] #Initialize list

for game in open ('games.txt', 'r'): # Reads a file and separates by rows, using game as an iterator
    words = game.split() # splits the iterator string within spaces
    
    city = ' '.join(words[:-1]) #merging strings
    year = words[:-1].strip('()') #removing round brackets from a string
    
    cities.append(city) #adding elements to the list 
    years.append(year)  #adding elements to the list 


# DERIVED ITERATORS 
 
for i,city in enumerate(cities[:10]): #loops through a list and creates tuples assigning numbers to the first 10 positions
    print (i, city)    
    
for city in sorted(cities[:10], key=len): #loops trough a sorted list and retruns the first 10 positions
    print(city)
    
for year,city in zip(cities[:10], years[:10]): #loops trough a joint tuple of lists and retruns the first 10 positions
    print(city)
    
    
# Comprehensions are a way to run a loop within a single line of code, and to collect the results of the loop in a collection, such as a list
# Generators are shortcuts to write functions that implement iterators.


import math, json, collections, itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import geopy

results = []

for city, year in zip(cities, years): #apply a change to every element in an iterator if the condition is satisfied 
    if int(year) > 1945:
        results.append(city + ': ' + year)

# now rewrite this in one line
        
results = [city + ': ' + year for city, year in zip(cities, years) if int(year) > 1945]
cities_by_year = {year: city for city, year in zip(cities, years) if int(year) > 1945}

# Sets are useful when we want to collect elements in a lsit only once

cities_after_1930 = {city for city, year in zip(cities, years) if int(year) > 1930}

gen = (i for i in range(20) if  i%2 ==0) # Generator
gen.__next__() # Iterator of a generator
sum(gen)

def fibonacci():
    f1, f2 = 0, 1
    
    while True:
        yield f2
        f1, f2 = f2, f1 + f2
        
f = fibonacci()

[next(f) for i in range(20)] #next(f) shorthand for f.__next__() 

# COLLECTIONS

import os as os 
import math, json, collections, itertools
import numpy as np
import matplotlib.pyplot as plt

 
os.chdir('C:\\Users\\clapsfi\\Desktop')

open('goldmedals.txt', 'r').readlines()[:10]

# Dictionaries are relatively memory-heavy objects

medal = collections.namedtuple('medal', ['year','athlete','team','event'])
m = medal('1896', 'Thomas Burke', 'USA', '100m')

medals = [medal(*line.strip().split('\t')) for line in open('goldmedals.txt', 'r')]

teams = collections.Counter(medal.team for medal in medals)

def best_by_year(year):
    counts = collections.Counter(medal.team for medal in medals if medal.year == str(year))
    best = counts.most_common(5)
    
    return[b[0] for b in best], [b[1] for b in best]
    
best_by_year(1900)

plt.style.use('ggplot')
colors = plt.cm.Set3(np.linspace(0,1,5))

def plotyear(year): 
    countries, tally = best_by_year(1900)

    bars = plt.bar(np.arange(5), tally, align='center')
    plt.xticks(np.arange(5), countries)
    
    for bar, color in zip(bars,colors):
        bar.set_color(color)
    
    plt.title(year)
    
plotyear(2016)

winners_by_country = {}

for medal in medals: 
    if medal.team not in winners_by_country:
        winners_by_country[medal.team] = [medal.athlete]
    else:
        winners_by_country[medal.team].append(medal.athlete)
        
winners_by_country['ITA']




winners_by_country = collections.defaultdict(list)

for medal in medals: 
    winners_by_country[medal.team].append(medal.athlete)

winners_by_country['ITA']


ordered_winners = collections.OrderedDict()

for medal in medals:
    if medal.team == 'ITA':
        ordered_winners[medal.athlete] = medal.event + ' ' + medal.year
        
ordered_winners

# also written
{medal.athlete: medal.event + ' ' + medal.year for medal in medals if medal.team == 'ITA'}

dq = collections.deque(range(10)) # Like a stack, where you can append and potp
dq

#The non-repetition code, seems to call for a SET collection

#ADDING ELEMENTS TO A NON-DUPLICATE LIST
athletes = sorted({medal.athlete for medal in medals})

#FINIDING COMMON ELEMENTS BETWEEN TWO LISTS

winners_100m = {medal.athlete for medal in medals if '100m' in medal.event} #in allows to compare full strings with partial strings! 
winners_200m = {medal.athlete for medal in medals if '200m' in medal.event}

vincitori = winners_100m & winners_200m

#STRING FORMATTING

for medal in medals:
    if medal.event.startswith('long jump'):
        print('In {0}, {1} won for {2}.'.format(medal.year, medal.athlete, medal.team))
        
def findmedal(**kwargs):
    return [medal for medal in medals if all(getattr(medal, key) == value for key,value in kwargs.items())]

findmedal(year='1896', team='USA')


#------------------------------------------------------------------------------
# FIND THE FIVE ATHLETES WHO WON THE MOST GOLD MEDALS
#------------------------------------------------------------------------------

import os as os 
import collections
import numpy as np
import matplotlib.pyplot as plt

 
os.chdir('C:\\Users\\clapsfi\\Desktop')

medal = collections.namedtuple('medal', ['year','athlete','team','event'])
medals = [medal(*line.strip().split('\t')) for line in open('goldmedals.txt', 'r')]

winners = collections.Counter(medal.athlete for medal in medals) # Counter counts how many times the name of an athlete appears
best = winners.most_common(5)

for b in best:
    print (b[0], b[1])
    
#------------------------------------------------------------------------------
# FIND THE FIVE ATHLETES WHO WON THE MOST GOLD MEDALS IN DIFFERENT EVENT
#------------------------------------------------------------------------------

events_by_athletes_set = collections.defaultdict(set) # Use this to avoid repetitions

def howmany(tup):
    return len(tup[1])

def clean(event):
    return ' '.join(word for word in event.split() if word not in ('men', 'women'))

for medal in medals:
    events_by_athletes_set[medal.athlete].add(clean(medal.event))


sorted(events_by_athletes_set.items(), key=howmany)
