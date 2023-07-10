###Created By: James Kyle Harrison
###Date: 04/25/2022
###Project Description: To take a dataset of metacritic user reviews and filter out fake reviews
and calculate an average of the real user reviews so that they can be compared to the average
of critic reviews described in another dataset. A dataset of training reviews is required to
create a bag of words to find words that are commonly in fake reviews to filter them.

###project.py: Used for training the bag of words, and creating a prediction on if a review
is real or fake using a Multinominal Naive Bayes approach. The predictions is exported back
to a csv.

###comparison.py: Takes in csv created by project.py and csv of metacritic critic reviews to
recalculate the average of user reviews and directly compare them to each other.
