#from IPython import embed
import numpy as np
import pandas as pd

data = pd.read_csv("Titanic_data.csv")
names = pd.read_csv("Titanic_names.csv")

new_data = data.merge(names, how='inner', on='id')

print(new_data.shape[0])
print(f"Amount of males: {new_data['Gender'].value_counts()['male']}")
print(f"Amount of females: {new_data['Gender'].value_counts()['female']}")
print(f"Average age of passengers: {new_data['Age'].mean().round(2)}")

#Noticed there were people with age of 0.xx. Guess in this dataset the 0 means "N/A".
print(f"Amount of 0 yeard olds: {new_data[new_data['Age']==0]['Age'].count()}")

#embed()

# Excercise 4
avg_age_over_0 = new_data[new_data['Age']>0]['Age'].mean().round(2)
print(f"Average age of passengers over 0: {avg_age_over_0}")
new_data['Age'] = new_data['Age'].replace(0, avg_age_over_0)

# Who has PClass "*"
print("\nWho has PClass of '*': \n")
print(new_data[new_data['PClass'].eq('*')])

#Surviors:

survivor_data = {
    'Survived': ['Yes', 'No'],
    'Count': [new_data['Survived'].value_counts()[1], new_data['Survived'].value_counts()[0]],
    'Percentage': [(new_data['Survived'].value_counts()[1]/new_data.shape[0]).round(2), (new_data['Survived'].value_counts()[0]/new_data.shape[0]).round(2)],
    'Male': [(new_data['Gender'].eq('male')&new_data['Survived'].eq(1)).value_counts()[True], (new_data['Gender'].eq('male')&new_data['Survived'].eq(0)).value_counts()[True]],
    '%OfAllMale': [((new_data['Gender'].eq('male')&new_data['Survived'].eq(1)).value_counts()[True]/new_data['Gender'].value_counts()['male']).round(2),
                ((new_data['Gender'].eq('male')&new_data['Survived'].eq(0)).value_counts()[True]/new_data['Gender'].value_counts()['male']).round(2)],
    '%OfSurvivorsMale': [((new_data['Gender'].eq('male')&new_data['Survived'].eq(1)).value_counts()[True]/new_data['Survived'].value_counts()[1]).round(2),
                ((new_data['Gender'].eq('male')&new_data['Survived'].eq(0)).value_counts()[True]/new_data['Survived'].value_counts()[0]).round(2)],
    'Female': [(new_data['Gender'].eq('female')&new_data['Survived'].eq(1)).value_counts()[True], (new_data['Gender'].eq('female')&new_data['Survived'].eq(0)).value_counts()[True]],
    '%OfAllFemale': [((new_data['Gender'].eq('female')&new_data['Survived'].eq(1)).value_counts()[True]/new_data['Gender'].value_counts()['female']).round(2),
                ((new_data['Gender'].eq('female')&new_data['Survived'].eq(0)).value_counts()[True]/new_data['Gender'].value_counts()['female']).round(2)],
    '%OfSurvivorsFemale': [((new_data['Gender'].eq('female')&new_data['Survived'].eq(1)).value_counts()[True]/new_data['Survived'].value_counts()[1]).round(2),
                ((new_data['Gender'].eq('female')&new_data['Survived'].eq(0)).value_counts()[True]/new_data['Survived'].value_counts()[0]).round(2)]}

survivor_data = pd.DataFrame(survivor_data,
                             columns = ['Survived', 'Count', 'Percentage', 'Male', '%OfAllMale', '%OfSurvivorsMale', 'Female', '%OfAllFemale', '%OfSurvivorsFemale'])

print(survivor_data)
