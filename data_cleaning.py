# -*- coding: utf-8 -*-
"""data_cleaning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qQH3uSOk6i-AwFuEKNV5KPYLwWAw-eqo
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)

def data_cleaning(df):
  df["concistency_check"] = np.zeros(df.shape[0])

  #Solving Missing Values Related to Education
  for edu in ['Brazil_education', 'Canada_edu', 'China_edu',
       'Ecuador_edu', 'France_edu', 'France_adjusted_edu', 'German_edu',
       'India_education', 'Italy_education', 'Mexico_edu', 'Nigeria_education',
       'Poland_education', 'Russian_education', 'South_Korea_edu',
       'Singapore_education', 'South_Africa_edu', 'Spain_education',
       'Sweden_education', 'UK_education', 'US_education']:

       df["concistency_check"] = df["concistency_check"] + df[edu].fillna(0)
  df = df.drop(columns="concistency_check")
  #The only problems of consistency are in France, as we have two variables corresponding to education in France: 'France_edu' and 'France_adjusted_edu'. 
  #The universial education measures takes only into account the 'France_adjusted_edu', which makes sense.
  #Then, we conclude that all those variables can be dropped with no loss of relevant information.
  #We will also rename the country column's values with the countries' names instead of their default numeric encoding.
  dic_idx_country = dict(enumerate(['Brazil', 'Canada', 'China', 'Ecuador', 
                                  'France', 'Germany', 'India', 'Italy',
                                  'Mexico', 'Nigeria', 'Poland', 'Russia',
                                  'South_Africa', 'Singapore', 'South_Korea',
                                  'Spain', 'Sweden', 'UK', 'US'], start=1))
  df["Country"].replace(dic_idx_country, inplace=True)
  df["Country2"].replace(dic_idx_country, inplace=True)
  df = df.drop(columns=['Brazil_education', 'Canada_edu', 'China_edu',
       'Ecuador_edu', 'France_edu', 'France_adjusted_edu', 'German_edu',
       'India_education', 'Italy_education', 'Mexico_edu', 'Nigeria_education',
       'Poland_education', 'Russian_education', 'South_Korea_edu',
       'Singapore_education', 'South_Africa_edu', 'Spain_education',
       'Sweden_education', 'UK_education', 'US_education'])
  #Solving Missing Values Related to Region
  df["Region"] = np.zeros(df.shape[0])
  for reg in df.filter(regex='.+_region'):
    df["Region"] = df["Region"] + df[reg].fillna(0)
  #As for the US_Ethnicity variable, although ethnicity can be an interesting information to analyses, the fact that is exlusive to the US and cannot be estimated makes it better to drop it.
  df = df.drop(columns=['Brazil_region', 'Canada_region', 'China_region', 'Ecuador_region',
       'France_region', 'German_regions', 'India_region', 'Italy_region',
       'Mexico_region', 'Nigeria_region', 'Poland_region', 'Russian_region',
       'South_Korea_region', 'Singapore_region', 'South_Africa_region',
       'Spain_region', 'Sweden_region', 'UK_region', 'US_region','US_Ethnicity'])
  #"Solving" Discrepancies and Missing Data Related to Income
  df = df.drop(columns=['Income', 'adjusted_income'])
  #Solving Discrepancies for Age: Some entries in the Age column are not numeric so we fix them
  df.loc[256, ['Age']] = 19
  df.loc[1433, ['Age']] = 30
  df.loc[2288, ['Age']] = (55+64)//2   # Age_group of the subject
  df.loc[3608, ['Age']] = 39
  df.loc[7898, ['Age']] = (55+64)//2   # Age_group of the subject
  df.loc[10648, ['Age']] = 28
  df.loc[10650, ['Age']] = 27
  df['Age'] = df['Age'].astype(int)
  #Solving Missing Values Related to Language
  freq_lang = df.groupby(['Country'])['Language'].agg(lambda x:x.value_counts(dropna= False).index[0]).to_dict()
  freq_lang['Mexico'] = 4.
  freq_lang['Singapore'] = 14
  freq_lang['South_Korea'] = 15.
  freq_lang['US'] = 1.
  for c in df['Country'].unique():
    df.loc[(df['Country']==c) & (pd.isnull(df['Language'])), 'Language'] = freq_lang[c]
  # Solving Unsatisfying Categorical Encoding for Gender : Following the reference paper, we know that 1 stands for ‘Male’ and 2 for ’Female’. 
  #However, there is no information on the meaning of 3 and 4. 
  #On can expect that this can be related to possible answers such as transgender, non-binary, rather not answer. To solve this problem, let aggroup these values as ‘Other’.
  df["Gender"].replace({1: "Male", 2: "Female", 3: "Other", 4: "Other"}, inplace=True)
  # Solving Unsatisfying Categorical Encoding for Presence of sick people in family
  #The variable sickwithCOVID has three categorical values, 1,2, and 3. Although paper 1 does not detail the meaning of each category, 
  #the provided code applies a logistic regression on each outome (Vaccine and Business2 discretized to two vategories, "low" and "high") given a
  #transformation of sickwithCOVID into a binary variable: 1 if sickwithCOVID==1, else 0 (i.e., if sickwithCOVID is 2 or 3). Looking at the table of results in paper 1, 
  #we may infer that this (transformed) variable corresponds to the survey question "Myself or family sick with COVID-19" of values "Yes" or "No",
  #with "Yes" apparently corresponding to (transformed) sickwithCOVID equal to 1 (because both the code and the paper show it has a positive impact on the targets,
  #though we could not reproduce the results of the paper with their code .
  #We will then apply the same transformation to that column.
  df['sickwithCOVID'] = df['sickwithCOVID'].transform(lambda x: 1 if (x == 1) else 0)
  #Solving hte Presence of Unknown Variables
  df = df.drop(columns=['Business1', 'Business3', 'Business4', 'Mode', 'vacc2'])
  #Further Analyses of Variables
  df = df.drop(columns=['Unique ID','Country2'])
  df = df.dropna()
  #Categorical features
  #Let us first convert columns whose values we already "renamed" (converting to a reasonable string) or will not rename.
  df['Country'] = df['Country'].astype('category')
  df['Gender'] = df['Gender'].astype('category')
  if 'Language' in df.columns:  # (avoid error if dropped)
      df['Language'] = df['Language'].astype('category')
  if 'Region' in df.columns:  # (avoid error if dropped)
      df['Region'] = df['Region'].astype('category')
  #Now, the columns that need renaming:
  df['Universal_edu'] = df['Universal_edu'].replace({1: "Less than high school", 2: "High school some college", 3: "Bachelor", 4: "Post Graduate"}).astype('category')
  df["within_country"] = df["within_country"].replace({1: "<20%", 2: "20-40%", 3: "40-60%", 4: "60-80%", 5:">80%", 6:"Refused"}).astype('category')
  df["world_wide"] = df["world_wide"].replace({1: "<$2 per day", 2: "\$2-\$8 per day", 3: "\$8-\$32 per day", 4: "$32+", 6:"Refused"}).astype('category')
  df["Age_group"] = df["Age_group"].replace({1: "18-24", 2: "25-54", 3: "55-64", 4: "65+"}).astype('category')

  return df