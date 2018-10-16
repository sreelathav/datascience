# Sree Latha Vallabhaneni      
#project under Strand 1: Statistical modelling/machine Learning
# Analysis of movie_metadata.csv  data, downloaded from kaggle website 
#https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset/

#Import libraries and required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

#Load data. clean, arrange or pick data variables in the useful way 
# and do exploratory analysis
df = pd.read_csv('Documents/dataProgPython/input/movie_metadata.csv')
df.head()
df[df.title_year > 2005].head()
#checking for null values
df.apply(lambda x: sum(x.isnull()),axis=0)

def create_comparison_database(name, value, x, no_films):
    
    comparison_df = df.groupby(name, as_index=False)
    
    if x == 'mean':
        comparison_df = comparison_df.mean()
    elif x == 'median':
        comparison_df = comparison_df.median()
    elif x == 'sum':
        comparison_df = comparison_df.sum() 
    
    # Create database with either name of directors or actors, the value being compared i.e. 'gross',
    # and number of films they're listed with. Then sort by value being compared.
    name_count_key = df[name].value_counts().to_dict()
    comparison_df['films'] = comparison_df[name].map(name_count_key)
    comparison_df.sort_values(value, ascending=False, inplace=True)
    comparison_df[name] = comparison_df[name].map(str) + " (" + comparison_df['films'].astype(str) + ")"
   
    # create a Series with the name as the index so it can be plotted to a subgrid
    comp_series = comparison_df[comparison_df['films'] >= no_films][[name, value]][10::-1].set_index(name).ix[:,0]
    
    return comp_series
    fig = plt.figure(figsize=(18,6))

# Director_name
plt.subplot2grid((2,3),(0,0), rowspan = 2)
create_comparison_database('director_name','gross','sum', 0).plot(kind='barh', color='#006600')
plt.legend().set_visible(False)
plt.title("Total Gross of Director's Films")
plt.ylabel("Director (no. films)")
plt.xlabel("Gross (in billons)")

plt.subplot2grid((2,3),(0,1), rowspan = 2)
create_comparison_database('director_name','imdb_score','median', 4).plot(kind='barh', color='#ffff00')
plt.legend().set_visible(False)
plt.title('Median IMDB Score for Directors with 4+ Films')
plt.ylabel("Director (no. films)")
plt.xlabel("IMDB Score")
plt.xlim(0,10)

plt.tight_layout()
fig = plt.figure(figsize=(18,6))

# Actor_1_name
plt.subplot2grid((2,3),(0,0), rowspan = 2)
create_comparison_database('actor_1_name','gross','sum', 0).plot(kind='barh', color='#006600', alpha=.8)
plt.legend().set_visible(False)
plt.title("Total Gross of Actor_1_name's Films")
plt.ylabel("Actor_1_name (no. films)")
plt.xlabel("Gross (in billons)")

plt.subplot2grid((2,3),(0,1), rowspan = 2)
create_comparison_database('actor_1_name','imdb_score','median', 4).plot(kind='barh', color='#ffff00', alpha=.8)
plt.legend().set_visible(False)
plt.title('Median IMDB Score for Actor_1_name with 8+ Films')
plt.ylabel("Actor_1_name (no. films)")
plt.xlabel("IMDB Score")
plt.xlim(0,10)

plt.tight_layout()
#############
#from sklearn.preprocessing import LabelEncoder
#var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
#le = LabelEncoder()
#for i in var_mod:
#    df[i] = le.fit_transform(df[i])
#df.dtypes 
##############
