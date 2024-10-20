import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from sklearn import svm 
import numpy as np



def get_clean_data():
    data = pd.read_csv('data/loan.csv')
    pd.set_option('future.no_silent_downcasting', True)
    data.dropna(inplace=True)
    data['Loan_Status'] = data['Loan_Status'].map({ 'Y': 1, 'N': 0 })
    #data.replace({'Loan_Status':{'Y':1, 'N':0}}, inplace= True)
    data.replace(to_replace='3+', value=4, inplace=True, )
    data.replace({'Married':{'Yes':1, 'No':0}, 'Gender':{'Male':1, 'Female':0}, 'Self_Employed':{'Yes':1, 'No':0},
                      'Property_Area':{'Rural':0, 'Semiurban': 1, 'Urban':2}, 'Education': {'Graduate':1,'Not Graduate':0}}, inplace=True)
    #print(data)

    return data

def create_model(data): 
    X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = data['Loan_Status']
    #print(y)
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=10, train_size=0.8
    )
    
    # train the model
    model = LogisticRegression()
    
    model = model.fit(X_train, y_train)
    
    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    input = (1,0,0,1,0,1853,2840,114,360,1,0) # Not Deserve 1,1,4,1,0,3036,2504,158,360,0,0
    #input = (1,1,1,1,0,4583,1508,128,360,1,0)
    input = np.asarray(input).reshape(1,-1)
    scaledinput = scaler.transform(input)
    print(scaledinput)
    a = model.predict(scaledinput)
    print('prediction is: ',a)
    f = list(input)
    w= f[0]
    print(w[1])
    if w[5] > w[6]:
        print('pred test greater')
    else:
        print('pred test less')
    if a == 1:
        print('Deserves a Loan (model file)')
    else:
        print('Does not deserve a loan (model file)')



    return model, scaler


def main():
    data = get_clean_data()
    
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__=='__main__':
    main()