import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

names = ['parents', 'h_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'target']
'''parents: usual, pretentious, great_pret
has_nurs: proper, less_proper, improper, critical, very_crit
form: complete, completed, incomplete, foster
children: 1, 2, 3, more
housing: convenient, less_conv, critical
finance: convenient, inconv
social: non-prob, slightly_prob, problematic
health: recommended, priority, not_recom
'''

df = pd.read_csv('nursery.data', sep=',', header=None)
print(df)
for i in range(len(names)):
    df[i] = df[i].astype('category')
df.columns = names
print(df)

map_parents = {'usual':3,'pretentious':2,'great_pret':1}
map_h_nurs={'proper':5, 'less_proper':4, 'improper':3, 'critical':2, 'very_crit':1 }
map_children={'1':1,'2':2,'3':3,'more':4}
map_housing={'convenient':3, 'less_conv':2, 'critical':1}
map_finance={ 'convenient':2, 'inconv':1}
map_social={'nonprob':3, 'slightly_prob':2, 'problematic':1}
map_health={'recommended':3, 'priority':2, 'not_recom':1}
map_form={'complete':1, 'completed':2, 'incomplete':3, 'foster':4}
df['parents'] = df['parents'].replace(map_parents)
df['h_nurs'] = df['h_nurs'].replace(map_h_nurs)
df['children'] = df['children'].replace(map_children)
df['housing'] = df['housing'].replace(map_housing)
df['finance'] = df['finance'].replace(map_finance)
df['social'] = df['social'].replace(map_social)
df['health'] = df['health'].replace(map_health)
df['form'] = df['form'].replace(map_form)
# df.iloc[:,2] = l_enc_x.fit_transform(df.iloc[:,2])
print(df)



scaler = MinMaxScaler().fit(df.iloc[:,:-1])
new_df = scaler.transform(df.iloc[:,:-1])
classes = df.iloc[:,-1]


'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
print(len(train))
print(len(test))
'''

#distance matrix using manhattan distance
d_matrix=np.zeros(len(new_df)**2).reshape(len(new_df),len(new_df))
for i in range(len(new_df)):
	for j in range(len(new_df)):
		d_matrix[i][i]=0
		dis=0
		for k in range(8):
			if(k==2):
				if(new_df[i][k]!=new_df[j][k]):
					dis+=1
			dis=dis+abs(new_df[i][k]-new_df[j][k])
		d_matrix[i][j]=dis/8

# print(d_matrix)
# f = open('dist_d','w+')
# for r in d_matrix:
# 	t = ''
# 	for s in r:
# 		t += str((s)) + ','
# 	f.write(t[:-1]+'\n')

# f.close()
