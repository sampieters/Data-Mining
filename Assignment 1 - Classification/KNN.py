import math
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.impute import KNNImputer

# suppress UserWarning for missing default style in the workbook
warnings.simplefilter("ignore", category=UserWarning)

#############################################
# PART 0: Create all instances used in the code
#############################################
# create an instance of the OrdinalEncoder
encoder = OrdinalEncoder()
# create a KNNimputer to fill missing values with a value close to it's neighbors
imputer = KNNImputer(n_neighbors=5)
scaler = MinMaxScaler()

#############################################
# PART 1: Training the data
#############################################
existing_customers = pd.read_excel('data/existing-customers.xlsx')
existing_customers = existing_customers.drop('RowID', axis=1)

# define the columns to encode
cols_to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# fit the encoder on the specified columns
encoder.fit(existing_customers[cols_to_encode])
# transform the specified columns
existing_customers[cols_to_encode] = encoder.transform(existing_customers[cols_to_encode])

data = existing_customers.drop(columns=['class'])
target = existing_customers['class']

# impute missing values in the categorical features
X_train_imputed = imputer.fit_transform(data)
# perform min-max scaling
scaled_data = scaler.fit_transform(X_train_imputed)
data = pd.DataFrame(scaled_data, columns=data.columns)

# split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data, target)

accuracies = []
best = 0
best_k = 0
for k in range(2, 200):
    # create a decision tree classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    # train the classifier on the training set
    clf.fit(X_train, Y_train)
    # predict the class labels for the testing set
    Y_pred = clf.predict(X_test)

    # evaluate the performance of the classifier using confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, labels=[">50K", "<=50K"])
    accuracy = accuracy_score(Y_test, Y_pred)

    # Get only the positives (true/false) because the negatives (true/false) only cost money to the business
    positives = (cm[0][0] + cm[1][0])
    # From these positives, calculate the amount of true positives and the amount of false positives based on the ratio from the confusion matrix
    true_positives = cm[0][0]
    false_positives = cm[1][0]
    send_costs = positives * (-10)
    true_responses = math.floor(true_positives * 0.10)
    false_responses = math.ceil(false_positives * 0.05)
    no_responses = positives - (true_responses + false_responses)
    profit = send_costs + (true_responses * 980) + (false_responses * -310)

    print(f'Classifier accuracy: {round(accuracy * 100, 2)}%    Tested total profit: {profit} euro')

    accuracies.append(profit)
    if profit > best:
        best = profit
        best_k = k

plt.plot(range(2, 200), accuracies)
plt.savefig('./data/k_values.png')
print(f'The best value of K is {best_k} with a profit of {best} euro')

#############################################
# PART 2: Apply the model on real data
#############################################
potential_customers = pd.read_excel('data/potential-customers.xlsx')
potential_customers = potential_customers.drop(columns=['RowID'])
# transform the specified columns
potential_customers[cols_to_encode] = encoder.transform(potential_customers[cols_to_encode])
potential_imputed = imputer.fit_transform(potential_customers)
scaled_data = scaler.fit_transform(potential_imputed)
potential_customers = pd.DataFrame(scaled_data, columns=potential_customers.columns)

# Make predictions
Y_test = clf.predict(potential_customers)

# Add the predictions to the potential customers dataset
potential_customers['class'] = Y_test

#############################################
# PART 3: Calculating the profit + listing the customers
#############################################

# Get only the positives (true/false) because the negatives (true/false) only cost money to the business
positives = potential_customers[potential_customers['class'] == '>50K']
# From these positives, calculate the amount of true positives and the amount of false positives based on the ratio from the confusion matrix
true_positives = round((cm[0][0] / (cm[0][0] + cm[1][0])) * len(positives))
false_positives = round((cm[1][0] / (cm[0][0] + cm[1][0])) * len(positives))

assert false_positives + true_positives == len(positives), "The sum of true and false positive does not match the amount of positives"

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