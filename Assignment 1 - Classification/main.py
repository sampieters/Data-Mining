import math
import warnings
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB, BernoulliNB, MultinomialNB

# suppress UserWarning for missing default style in the workbook
warnings.simplefilter("ignore", category=UserWarning)

#############################################
# PART 0: Create all instances used in the code
#############################################
encoder = OrdinalEncoder()
imputer = KNNImputer()
scaler = MinMaxScaler()
clf = RandomForestClassifier()

#############################################
# PART 1: Training the data
#############################################
existing_customers = pd.read_excel('data/existing-customers.xlsx')
existing_customers = existing_customers.drop('RowID', axis=1)

# define the columns to encode from strings to float values, fit the encoder to these columns, and convert the values
cols_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
encoder.fit(existing_customers[cols_to_encode])
existing_customers[cols_to_encode] = encoder.transform(existing_customers[cols_to_encode])

data = existing_customers.drop(columns=['class'])
target = existing_customers['class']

# impute missing values in the categorical features, here the MinMaxScaler which scales the values between 0 and 1.
X_train_imputed = imputer.fit_transform(data)
scaled_data = scaler.fit_transform(X_train_imputed)
data = pd.DataFrame(scaled_data, columns=data.columns)

# split the dataset into training and testing sets, train the classifier on the training set, and predict the class
# labels for the testing set.
X_train, X_test, Y_train, Y_test = train_test_split(data, target)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# evaluate the performance of the classifier using confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=[">50K", "<=50K"])
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Classifier accuracy: {round(accuracy * 100, 2)}%')

# Get only the positives (T/F) because the negatives (T/F) only cost money to the business and calculate the profit.
send_costs = (cm[0][0] + cm[1][0]) * (-10)
true_responses = math.floor(cm[0][0] * 0.10)
false_responses = math.ceil(cm[1][0] * 0.05)
no_responses = (cm[0][0] + cm[1][0]) - (true_responses + false_responses)
profit = send_costs + (true_responses * 980) + (false_responses * -310)
print(f"The tested total profit is: {profit} euro\n")

#############################################
# PART 2: Apply the model on real data
#############################################
potential_customers = pd.read_excel('data/potential-customers.xlsx')
potential_customers = potential_customers.drop(columns=['RowID'], axis=1)
# transform the specified columns in the same ways as with the exisitng customers
potential_customers[cols_to_encode] = encoder.transform(potential_customers[cols_to_encode])
potential_imputed = imputer.fit_transform(potential_customers)
scaled_data = scaler.fit_transform(potential_imputed)
potential_customers = pd.DataFrame(scaled_data, columns=potential_customers.columns)

# Make predictions and add them to the potential customers dataset
potential_customers['class'] = clf.predict(potential_customers)

#############################################
# PART 3: Calculating the profit + listing the customers
#############################################
# Get only the positives (true/false) because the negatives (true/false) only cost money to the business
positives = potential_customers[potential_customers['class'] == '>50K']
# From these positives, calculate the amount of true positives and the amount of false positives based on the ratio from the confusion matrix
true_positives = round((cm[0][0] / (cm[0][0] + cm[1][0])) * len(positives))
false_positives = round((cm[1][0] / (cm[0][0] + cm[1][0])) * len(positives))

# Save the row IDs of selected customers to a file
file = open("data/selection.txt", "w")
for index in positives.index:
      file.write(f"Row ID: {index}\n")
file.close()

send_costs = len(positives) * (-10)
true_responses = math.floor(true_positives * 0.10)
false_responses = math.ceil(false_positives * 0.05)
no_responses = len(positives) - (true_responses + false_responses)
profit = send_costs + (true_responses * 980) + (false_responses * -310)

print(f"There are {len(positives)} people send an e-mail (see \"./data/collected.txt)\"\n"
      f"{true_responses} people with >50K respond making/losing {true_responses * (980 - 10)} euro \n"
      f"{false_responses} people with <=50K respond making/losing {false_responses * (-310 - 10)} euro \n"
      f"{no_responses} people do not respond which loses {no_responses * (-10)} euro \n"
      f"The total profit is: {profit} euro")