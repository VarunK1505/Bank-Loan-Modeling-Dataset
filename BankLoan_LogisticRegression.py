import numpy as np
import pandas as pd

#references https://www.youtube.com/watch?v=JDU3AzH3WKg

class LogisticRegression:
    
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr;
        self.n_iters = n_iters;
        self.weights = None;
        self.bias = None;
    
    def fit(self, X, y):
        # initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0;
        
        #gradiant descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias; #y = ax + b
            y_predicted = self._sigmoid(linear_model)
            
            #update rule w = w - a.dw , b = b - a.db (a = learning rate, d = derivative)
            #J'(thetha) = [dJ/dw] = [1/N(sum(2x(y' - y)))]   //y' = predicted y
            #             [dJ/db]   [1/N(sum(2(y'-y)))]
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw;
            self.bias -= self.lr*db;
        
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias;
        y_predicted = self._sigmoid(linear_model)
        # more than 0.5 then class 1 else class 0
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted];
        
        return y_predicted_cls;
    
    def _sigmoid(self, x):
        #helper function, y = 1/(1+e^(-x))
        
        return 1/(1 + np.exp(-x))

#Loading our Data
file_loc = "Bank_Personal_Loan_Modelling.csv"
bank_data = pd.read_csv(file_loc);

#Data Preprocessing
bank_data.set_index("ID",inplace=True);

bank_data.drop('ZIP Code',axis=1,inplace=True)

bank_data.drop('Experience',axis=1,inplace=True)

bank_data.drop('CCAvg',axis=1,inplace=True)

y = bank_data['Personal Loan'];

bank_data.drop('Personal Loan', axis=1,inplace=True)

X = bank_data;

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4);

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

classifier = LogisticRegression(lr = 0.00001, n_iters = 1000)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print("Logistic Regression classification accuracy is: ", accuracy(y_test, predictions)*100);
