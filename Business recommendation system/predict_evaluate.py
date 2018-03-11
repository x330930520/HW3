import pandas as pd
import numpy as np
import sys
import sys

def main(city):
    # Memory-based CF prediction
    def demean(train, mean):
        train_demean = train.as_matrix()
        x, y = train.shape
        for i in range(x):
            m = mean[i]
            for j in range(y):
                if train_demean[i,j] != 0:
                    train_demean[i,j] -= m
        return train_demean

    def memory_predict(train, similarity, type):
        if type == 'user':
            # Compute the average score each user gives
            sum = train.sum(axis=1)
            non_zero = train.astype(bool).sum(axis=1)
            mean = (sum.iloc[:]/non_zero.iloc[:]).fillna(0).values
            # Compute demeaned data
            train_demean = demean(train, mean)
            raw_predict = np.dot(similarity, train_demean)
            normalizer = np.abs(similarity.sum(axis=1).values)
            predict = (raw_predict.T/normalizer + mean).T
        elif type == 'business':
            raw_predict = np.dot(train, similarity)
            normalizer = np.abs(similarity.sum(axis=1).values)
            predict = raw_predict / normalizer
        else:
            print("Type not recognised")
        return predict

    # Model-based CF prediction
    from scipy.sparse.linalg import svds
    def model_predict(train):
        sum = train.sum(axis=1)
        non_zero = train.astype(bool).sum(axis=1)
        mean = (sum.iloc[:]/non_zero.iloc[:]).fillna(0).values
        train_demean = demean(train, mean)
        u, s, vt = svds(train_demean, k=50)
        s_diag_matrix = np.diag(s)
        predict = np.dot(np.dot(u, s_diag_matrix), vt)
        for i in range(predict.shape[0]):
            predict[i, :] += mean[i]
        return predict

    # Evaluation the results
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    def rmse(predict, test):
        prediction = predict[test.nonzero()].flatten()
        ground_truth = test[test.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    def mae(predict, test):
        prediction = predict[test.nonzero()].flatten()
        ground_truth = test[test.nonzero()].flatten()
        return sum(abs(prediction - ground_truth)) / len(prediction)

    # Import the training and testing data
    train = pd.read_pickle(city + '_train_review.pkl')
    test = pd.read_pickle(city + '_test_review.pkl')
    print("Begin to predict and evaluate")

    # Train a memory-based CF and evaluate it
    user_sim = pd.read_pickle(city + '_user_similarity.pkl')
    user_predict = memory_predict(train, user_sim, 'user')
    user_evaluation = rmse(user_predict, test.iloc[:,:].values)
    print("User-based CF RMSE: " + str(user_evaluation))
    user_evaluation = mae(user_predict, test.iloc[:,:].values)
    print("User-based CF MAE: " + str(user_evaluation))

    train = pd.read_pickle(city + '_train_review.pkl')
    business_sim = pd.read_pickle(city + '_business_similarity.pkl')
    business_predict = memory_predict(train, business_sim, 'business')
    business_evaluation = rmse(business_predict, test.iloc[:,:].values)
    print("Item-based CF RMSE: " + str(business_evaluation))
    business_evaluation = mae(business_predict, test.iloc[:,:].values)
    print("Item-based CF MAE: " + str(business_evaluation))

    # Train a model-based CF and evaluate it
    train = pd.read_pickle(city + '_train_review.pkl')
    model_predict = model_predict(train)
    model_evaluation = rmse(model_predict, test.iloc[:,:].values)
    print("Model-based CF RMSE: " + str(model_evaluation))
    model_evaluation = mae(model_predict, test.iloc[:,:].values)
    print("Model-based CF MAE: " + str(model_evaluation))



'''
user_mat = np.zeros((user_num, user_num))-1
for i in range(user_num):
    for j in range(user_num):
        if user_mat[j][i] == -1:
            user_mat[i][j] = 1- cosine(train.iloc[i, :].values,
                                       train.iloc[j, :].values)
        else:
            user_mat[i][j] = user_mat[j][i]
    print("Training for user similarity: on stage "
          + str(i+1) + ' of ' + str(user_num))
print("finished user similarity")

business_mat = np.zeros((business_num, business_num))
for i in range(business_num):
    for j in range(business_num):
        if business_mat[j][i] == -1:
            business_mat[i][j] = 1 - cosine(train.iloc[:, i].values,
                                            train.iloc[:, j].values)
        else:
            business_mat[i][j] = business_mat[j][i]
    print("Training for business similarity: on stage "
          + str(i+1) + ' of ' + str(business_num))
print("finished business similarity")

# Base case: random
    base_predict = np.random.randint(6, size=train.shape)
    base_evaluation = rmse(base_predict, test.iloc[:,:].values)
    print("Random guess RMSE: " + str(base_evaluation))
    base_evaluation = mae(base_predict, test.iloc[:,:].values)
    print("Random guess MAE: " + str(base_evaluation))

'''

