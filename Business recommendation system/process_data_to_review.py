import json
import pandas as pd
import numpy as np
import sys

def main(city, user_threshold, business_threshold):

    # Get data from business.json
    business = []
    with open('business.json', encoding = 'utf8') as f:
        for line in f.readlines():
            cty = json.loads(line)['city']
            if cty == city:
                cnt = json.loads(line)['business_id']
                tmp = [cnt]
                business.append(tmp)
    business = pd.DataFrame(business, columns = ['business_id'])
    print("There are " + str(business.shape[0]) +
          " businesses in " + city)
    print("finished business.json")

    # Get data from user.json
    user = []
    with open('user.json', encoding = 'utf8') as f:
        for line in f.readlines():
            cnt = json.loads(line)['review_count']
            if cnt > user_threshold:
                u_id = json.loads(line)['user_id']
                tmp = [u_id, cnt]
                user.append(tmp)
    user = pd.DataFrame(user, columns = ['user_id', 'review_count'])
    print("There are " + str(user.shape[0]) +
          " users with more than " + str(user_threshold) + " reviews.")
    print("finished user.json")

    # Get data from review.json
    review = []
    with open('review.json', encoding = 'utf8') as f:
        user_dict = {}
        for k in user['user_id'].values:
            user_dict[k] = 0
        business_dict = {}
        for k in business['business_id'].values:
            business_dict[k] = 0
        for line in f.readlines():
            u_id = json.loads(line)['user_id']
            b_id = json.loads(line)['business_id']
            if u_id in user_dict and b_id in business_dict:
                st = json.loads(line)['stars']
                tmp = [u_id, b_id, st]
                review.append(tmp)
                user_dict[u_id] += 1
                business_dict[b_id] += 1
        initial_review = pd.DataFrame(review, columns = ['user_id', 'business_id', 'stars'])
        print("Initial review created")
        print("Size of review now: " + str(initial_review.shape[0]))

        # Check if the user review count > threshold
        names = ['user_id', 'business_id', 'stars']
        review_list = []
        for key, value in user_dict.items():
            if value >= user_threshold:
                tmp = initial_review[initial_review.user_id == key]
                review_list.append(tmp)
        review = pd.concat(review_list)
        print("Cleaned review data based on user review threshold")
        print("Size of review now: " + str(review.shape[0]))
        # Check if the business review count > threshold
        review_list = []
        for key, value in business_dict.items():
            if value >= business_threshold:
                tmp = review[review.business_id == key]
                review_list.append(tmp)
        review = pd.concat(review_list)
        print("Cleaned review data based on business review threshold")
        print("Size of review now: " + str(review.shape[0]))

    filename = city + '_review.pkl'
    review.to_pickle(filename)
    print("Stored review data to " + filename)

    # Store review data in matrix form
    def store_review_data_in_matrix(review, filename):
        mat = np.zeros((user_num, business_num))
        review_num = review.shape[0]

        # Create dictionary to store the ratings
        dict = {}
        for i in range(review_num):
            uid = review.iloc[i]['user_id']
            bid = review.iloc[i]['business_id']
            rate = review.iloc[i]['stars']
            dict[uid+bid] = rate

        # Add the values into the matrix
        for i in range(len(u)):
            uid = u[i]
            for j in range(len(b)):
                bid = b[j]
                if uid+bid in dict:
                    mat[i][j] = dict[uid+bid]
                j += 1

        user_business = pd.DataFrame(mat, index = u, columns = b)
        user_business.to_pickle(filename + '.pkl')
        print("Store matrix-form review data in " + filename + '.pkl')

    from sklearn.model_selection import train_test_split
    u = pd.unique(review['user_id'].values)
    b = pd.unique(review['business_id'].values)

    user_num = len(u)
    business_num = len(b)
    review_num = review.shape[0]
    print("number of user: " + str(user_num))
    print("number of business: " + str(business_num))
    print("number of review: " + str(review_num))
    if user_num*business_num != 0:
        print("sparsity of the matrix: "
              + format(100*review_num/(user_num*business_num), '0.5f')
              + '%')

    # Split the data
    train, test = train_test_split(review, test_size=0.25)

    # Store data in matrix form
    store_review_data_in_matrix(train, city + '_train_review')
    store_review_data_in_matrix(test, city + '_test_review')


'''
business = []
with open('business.json') as f:
    for line in f.readlines():
        print(json.load(line)['business_id'])
        business.append(json.loads(line))
business_keys = [x for x in business[0].keys()]

lst = []
for i in range(len(business)):
    lst.append([business[i][x] for x in business_keys])

business_df = pd.DataFrame(lst, columns = business_keys)

b = pd.unique(business_df['city'])
print(b)
print("There are " + str(len(b)) + " cities in the dataset.")
print(len(business_df[business_df['city'] == 'Toronto']))

city = 'Toronto'
cleaned_business = business_df[business_df['city'] == city][['business_id','state']].reset_index(drop=True)

cleaned_business.to_pickle('cleaned_business.pkl')
print("finished bussiness.jason and stored the business data")
'''

'''
user = []
with open('user.json') as f:
    for line in f.readlines():
        user.append(json.loads(line))
user_keys = [x for x in user[0].keys()]
lst = []
for i in range(len(user)):
    lst.append([user[i][x] for x in user_keys])
user_df = pd.DataFrame(lst, columns = user_keys)
threshold = 40
cleaned_user_df = user_df[user_df['review_count'] >= threshold][['user_id', 'review_count', 'average_stars']].reset_index(drop=True)
cleaned_user_df.to_pickle('cleaned_user.pkl')
print("finished user.jason and stored the user data")
'''

'''
cleaned_user = pd.read_pickle('cleaned_user.pkl')
cleaned_business = pd.read_pickle('cleaned_business.pkl')

review = []
with open('review.json', encoding = 'utf8') as f:
    for line in f.readlines():
        review.append(json.loads(line))
review_keys = [x for x in review[0].keys()]

lst = []
for i in range(len(review)):
    lst.append([review[i][x] for x in review_keys])

review_df = pd.DataFrame(lst, columns = review_keys)

cleaned_review_df = review_df[review_df['user_id'].isin(cleaned_user['user_id'].values) & review_df['business_id'].isin(cleaned_business['business_id'].values)][['review_id', 'user_id', 'business_id', 'stars']].reset_index(drop=True)

cleaned_review_df.to_pickle('cleaned_review.pkl')
print("finished review.jason and stored the review data")
'''

