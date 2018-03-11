import process_data_to_review
import train_similarity
import predict_evaluate
import sys

argv = sys.argv
length = len(argv)

if length == 1:
    print("please indicate a city")
elif length >= 2:
    city = sys.argv[1]
    user_threshold = int(sys.argv[2])
    business_threshold = int(sys.argv[3])
    # extract review data based on city
    # convert review data to train and test
    print("Begin to precess the data")
    process_data_to_review.main(city, user_threshold, business_threshold)
    print()
    # train similarity matrix using train
    print("Begin to train similarity")
    train_similarity.main(city)
    print()
    # conduct prediction and evaluation
    print("Begin prediction and evaluation")
    predict_evaluate.main(city)
