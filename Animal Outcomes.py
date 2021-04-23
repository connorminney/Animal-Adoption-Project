
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

#====================================================================================================#

'''Repeat with categorical conversion '''

data = pd.read_csv('PATH\\TO\\FILE\\aac_intakes_outcomes.csv')

data = data[['outcome_type', 'sex_upon_outcome',  'age_upon_outcome_(years)', 'outcome_month', 'outcome_year', 'animal_type', 'breed', 'color', 'intake_condition', 'intake_type', 'age_upon_intake_(years)', 'sex_upon_intake', 'intake_month', 'intake_year', 'time_in_shelter_days']]

# Split into dogs only
data = data[data['animal_type'] == 'Dog']
data.drop('animal_type', axis = 1, inplace = True)

# Convert outcomes to binary variables - only use adoption and Euthanasia
data = data[data['outcome_type'].isin(['Adoption','Euthanasia'])]
data['outcome_type'] = data['outcome_type'].str.replace('Euthanasia', '1')
data.loc[data['outcome_type'] != '1', 'outcome_type'] = 0
data['outcome_type'] = data['outcome_type'].astype(int)

# Split the sex columns 
data[['intake_spayed_neutered','sex']] = data['sex_upon_intake'].str.split(' ', 1, expand=True)
data.drop('sex_upon_intake', axis = 1, inplace = True)
data[['outcome_spayed_neutered','sex']] = data['sex_upon_outcome'].str.split(' ', 1, expand=True)
data.drop('sex_upon_outcome', axis = 1, inplace = True)

# Convert sex and spayed/neutered status
data = data[data['intake_spayed_neutered'] != 'Unknown'].dropna()
data = data[data['outcome_spayed_neutered'] != 'Unknown'].dropna()
data['intake_spayed_neutered'] = data['intake_spayed_neutered'].str.replace('Neutered','1').replace('Spayed','1').replace('Intact','0').astype(int)
data['outcome_spayed_neutered'] = data['outcome_spayed_neutered'].str.replace('Neutered','1').replace('Spayed','1').replace('Intact','0').astype(int)
data['sex'] = data['sex'].str.replace('Male','1').replace('Female','0').astype(int)

# Create a variable to determine if the dog is a mix
data['Mix'] = data['breed'].str.contains('Mix').astype(int)
data['breed'] = data['breed'].str.replace('Mix','')

# Reindex the data
data.reset_index(inplace = True, drop = True)
       
# Convert categorical to numeric
text_cols = ['breed', 'color', 'intake_condition', 'intake_type']

for i in text_cols:
    dummies = pd.get_dummies(data[i])
    data = data.merge(dummies, left_index = True, right_index = True)
    data.drop(i, axis = 1, inplace = True)
    

# Scale the age variable
#scaler = MinMaxScaler()
#data['intake_age_scaled'] = scaler.fit_transform(data['age_upon_intake_(years)'].values.reshape(-1,1))
#data['outcome_age_scaled'] = scaler.fit_transform(data['age_upon_outcome_(years)'].values.reshape(-1,1))

# Check correlations
correlations2 = data.corrwith(data["outcome_type"])

# Select variables
independent = ['outcome_spayed_neutered', 'Normal', 'Euthanasia Request', 'age_upon_intake_(years)', 'Pit Bull']
x = data[independent]
y = data['outcome_type']

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8)


#====================================================================================================#
''' Because the data is skewed toward adoptions, we you may want to balance the inputs to produce an accurate model '''

# Concatenate the x and y training sets that we just split
# train_data = pd.concat([x_train, y_train], axis = 1)

# # Split the recombined data into adoption/euthanasia sets
# adopted = train_data[train_data['outcome_type'] == 0]
# euthanized = train_data[train_data['outcome_type'] == 1]

# # Upsample the number of euthanasia records to match the length of the adoption records
# pop_upsampled = resample(adopted, replace = True, n_samples = len(euthanized))

# # Merge into one dataset
# data = pd.concat([euthanized, pop_upsampled])

# # Split back into dependent/independent variables
# x = data[independent]
# y = data['outcome_type']

# # Split the new dataset into training and test sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .8)

#====================================================================================================#

# Format the model inputs as numpy arrays
x_train = x_train.values
y_train = y_train.values.reshape(-1,1)

# Define a ML Model - can be anything
model = RandomForestClassifier(n_estimators = 100)

# Compute the RMSE and R2
scores = cross_val_score(model, x, y, cv = 10, scoring = 'accuracy')
r2_scores = cross_val_score(model, x, y, cv = 10, scoring = 'r2')
accuracy = scores.mean()
r2 = r2_scores.mean()

# Fit and predict the outcomes
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# Check the results
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('\n', 'Model R2: {}'.format(r2))
print('Model Accuracy: {}'.format(accuracy), '\n')
