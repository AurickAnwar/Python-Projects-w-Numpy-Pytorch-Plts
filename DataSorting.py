import pandas as pd
import matplotlib.pyplot as plt
#df = pd.read_csv('bd-dec22-births-deaths-natural-increase.csv')

#Data Cleaning and Visualization
#df = pd.DataFrame()
#series = pd.Series()
#print(df)
#print(series)

#animals = pd.Series(('dog', 'cat', 'fish', 'spider'))
#legs = pd.Series((4,4,0,8))
#print(animals)
#print(legs)

#animals_legs = pd.DataFrame({"Animal": animals, "Legs": legs})
#animals_legs.head()
#animals_legs.drop(0, axis=0, inplace = True)#drops 0 from the axis
#animals_legs.drop(["Animal"], axis = 1, inplace=True)#Drops animals from the data frame
#print(animals_legs.head())

df = pd.read_csv('california_housing_train.csv')
#df.drop(0, axis = 0, inplace=True)
#df.drop(['median_income'], axis = 1, inplace=True)
#df.drop(['median_house_value'], axis = 1, inplace=True)#drops columns
#pop_over_500 = df[df['population']>500]

#coordinates = df[['longitude','latitude']]#prints only the ones I say
#housing = df[['households']]#prints just housing
#lotitude = df[(df['latitude']>40) & (df['longitude']<-114)]#prints only the ones in a certain array

#exercise 2
#Create a new column in df that determines the average number of people per household on each block.
#From your new column, categorize the house according to the following rules:
#if avg number of people per household between 0 and 2 (inclusive): couple
#if avg number of people per household between 3 and 5: family
#if avg number of people per household greater than 5: tenants
#Give the mean of the median_house_value and median_house_income for each category


df['Average people per household'] = round(df['population']/df['households'])

def avg_per_household(temp_df):
   if temp_df['Average people per household']>=0 and temp_df['Average people per household']<=2:
       return "Couple"
   elif temp_df['Average people per household']>=3 and temp_df['Average people per household']<=5:
        return "Family"
   elif temp_df['Average people per household']>=5:
        return "Tenants"

df['Average people per household'] = df.apply(avg_per_household, axis=1)
filtered_df = df[(df['population']>1500) & (df['households']<400)]
avg = filtered_df[['population', 'households', 'Average people per household']]

df.drop(0, axis=0, inplace=True)

df.head()
print(avg)

