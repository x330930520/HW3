from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import numpy as np
import sys

def main(city):
    # Build similarity matrix
    train = pd.read_pickle(city + '_train_review.pkl')
    user_num, business_num = train.shape
    print("In training data, the number of users is "
          + str(user_num) +
          ", the number of businesses is "
          + str(business_num))

    user_mat = 1 - pairwise_distances(train, metric='cosine')
    print("finish user similarity")
    user_sim = pd.DataFrame(user_mat)
    user_sim.to_pickle(city + '_user_similarity.pkl')
    print("Stored user similarity in file "
          + city + '_user_similarity.pkl')

    business_mat = 1 - pairwise_distances(train.T, metric='cosine')
    print("finish business similarity")
    business_sim = pd.DataFrame(business_mat)
    business_sim.to_pickle(city + '_business_similarity.pkl')
    print("Stored business similarity in file "
          + city + '_business_similarity.pkl')


