#!/usr/bin/env python
# coding: utf-8

# # Revision of Dataframe

# ## Dataframe from the Previous model

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('df.csv')
print("df.shape",df.shape)
item_count = df["target"].value_counts()
print("Number of each species in dataframe:\n",item_count)


# ## Create New Dataframe

# In[2]:


"""
Creating individual dataframe of 'A_costaricaensis' , 'A_neoniger' , 'A_tubingensis', 'A_niger' and 'A_welwitschiae'
"""
df_class6 = df[df['target'] == 'A_costaricaensis']
df_class16 = df[df['target'] == 'A_neoniger']
df_class22 = df[df['target'] == 'A_tubingensis']
df_class17 = df[df['target'] == 'A_niger']
df_class25 = df[df['target'] == 'A_welwitschiae']


"""
Select 80 data randomly from each dataframe of 'A_costaricaensis' , 'A_neoniger' and 'A_tubingensis'
Select 140 data randomly from each dataframe of 'A_niger' and 'A_welwitschiae'
(the number 80,140 based on the minimum of data in each group)
"""
df_class6_rd = df_class6.sample(n = 80)
df_class16_rd = df_class16.sample(n = 80)
df_class22_rd = df_class22.sample(n = 80)
df_class17_rd = df_class17.sample(n = 140)
df_class25_rd = df_class25.sample(n = 140)


"""
Selecting all data from the dataframe in which ‘target’ is not 
'A_costaricaensis','A_neoniger','A_tubingensis', 'A_niger' and 'A_welwitschiae'
"""
target_cut = ['A_costaricaensis','A_neoniger','A_tubingensis','A_niger','A_welwitschiae']
df_cut = df.loc[~df['target'].isin(target_cut)]
print("df after cut 5 species out.shape",df_cut.shape)


"""
Concatenate dataframe (samples randomly) of 
'A_costaricaensis','A_neoniger','A_tubingensis','A_niger','A_welwitschiae'
"""
list_5 = [df_class6_rd,df_class16_rd,df_class22_rd,df_class17_rd,df_class25_rd]
df_com5 = pd.concat(list_5, axis=0, ignore_index=True)
print("\nShape of combined 5 species dataframe :",df_com5.shape)
item_counts_5 = df_com5["target"].value_counts()
print("Number of each species :\n",item_counts_5)


"""
Renamed A_costaricaensis and A_neoniger in combined dataframe as A_tubingensis 
and renamed A_welwitschiae as A_niger

since A_costaricaensis and A_neoniger are synnonyms of A_tubingensis
and A_welwitschiae is a synnonyms of A_niger 
(Bian et al. 2022)
"""
df_com5_rename = df_com5.replace({'A_costaricaensis':'A_tubingensis',
                                  'A_neoniger':'A_tubingensis',
                                  'A_welwitschiae':'A_niger'})
print("\nShape of renamed dataframe :",df_com5_rename.shape)
print

item_counts = df_com5_rename["target"].value_counts()
print("Number of each species :\n",item_counts)


"""
Combine dataframe of cut dataframe and renamed dataframe
"""
list_2 = [df_cut,df_com5_rename]
df_all = pd.concat(list_2, axis=0, ignore_index=True)
df_all.to_csv('dataframe.csv', index=False)  # Save to csv file
print("\nSave file as: dataframe.csv")
print("Shape of dataframe :",df_all.shape)

print("Number of each species in dataframe:\n",df_all["target"].value_counts())


# ## Count no. of member in each set & Print class mapping encoder  

# In[3]:


cols =[x for x in df.columns if x not in ['target']]
rowused = []
for i in range (len(df)):
        if i % 10 == 0:
            rowused.append('test')
                
        elif i % 10 == 1:
            rowused.append('validate')
                
        else:
            rowused.append('train')
                            
df['rowused'] = rowused
dd=df['rowused'].sample(len(df))
test_set=df[df['rowused']=='test']
validate_set=df[df['rowused']=='validate']
train_set=df[df['rowused']=='train']
print('Count test_set:\n',test_set['target'].value_counts())
print('\nCount validate_set:\n',validate_set['target'].value_counts())
print('\nCount train_set:\n',train_set['target'].value_counts())
print('------------------------------------------')
label_encoder = LabelEncoder()
data_y = df.loc[:, 'target']
encoded_y = label_encoder.fit_transform(data_y.values.ravel())
label_encoder_name_mapping = dict(zip(label_encoder.classes_,label_encoder.transform(label_encoder.classes_)))
print('Mapping of Label Encoded Classes:', label_encoder_name_mapping, sep="\n")

