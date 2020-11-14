# Import CSV and Pandas for this quiz.
import pandas as pd
import csv

# Import knn as our machine learning model. 
from sklearn.neighbors import KNeighborsClassifier

# Convert the csv files into dataframes.
classes = pd.read_csv('animal_classes.csv')
animals_data_train = pd.read_csv('animals_train.csv')
animals_data_test = pd.read_csv('animals_test.csv')

data_train_target = animals_data_train.class_number.values.reshape(-1,1)
print(data_train_target)

data_train = animals_data_train.drop('class_number',axis=1)

data_train = data_train.values

data_test = animals_data_test.drop('animal_name',axis=1)

data_test = data_test.values

knn = KNeighborsClassifier()

knn.fit(X=data_train,y=data_train_target)

predict_class_type = knn.predict(data_test)

animals_class_types = []

for c in predict_class_type:
    animals_class_types.append(classes.iat[(c - 1),2])

print(animals_class_types)

animal_class_match = []

for a in animals_data_test.animal_name.values:
    animal_class_match.append(a)

print(animal_class_match)

animal_dict = {'animal_name':animal_class_match, 'prediction':animals_class_types}
#print(animal_dict)

prediction_df = pd.DataFrame(animal_dict)
print(prediction_df)

prediction_df.to_csv(r'predictions.csv', index=False)