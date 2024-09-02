import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle 

#Импорт данных. Колонки:
'''
species - вид
island - остров обитания
bill_length_mm - длина клюва по оси Х
bill_depth_mm - висота ключа по оси У
flipper_length_mm - длина крыла
body_mass_g - вес
sex - пол
'''
penguins = pd.read_csv('.\data\penguins_cleaned.csv')
df = penguins.copy()
#Предсказываем вид пингвина. Predict species penguins
target = 'species'      #Если захотим предсказывать пол пингвина, поменяем местаи sex в target, и species в encode местами
encode = ['sex','island']

#Ordinal features encoding
for col in encode:
    #Convert categorical variable into dummy(фиктивная)/indicator variables
    #
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1) #axis=1 - столбец
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode) 

#Выделяем тренировочный датасет без целевой переменной
# (X - input, Y - output)
X = df.drop('species',axis=1)
#Целевая переменная. Наш таргет
Y = df['species']

#Классификатор
clf = RandomForestClassifier()
clf.fit(X,Y)


#Сохраняем модель
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))