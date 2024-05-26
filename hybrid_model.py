# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:56:42 2022

@author: Serena, Nikhil, Dhairya
"""

# importing the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# downloadin the stopwords
nltk.download('stopwords')

# downloading the lemmtizer
nltk.download('omw-1.4')

# importing dataframe
data = pd.read_csv('data/follower_of_top_10_stremers_pivot_more_than_4_streamers.csv')
del data["following_count"]


#############################################################################################################################
# COLLABORATIVE FILTERING
#############################################################################################################################

# building item-item collaborative filtering
similarity_df = pd.DataFrame(np.zeros((99,99), dtype=float), columns=data.columns)

# updating the indexes 
similarity_df.index = data.columns

# function to compute jaccard similarity
def jaccard_similarity(x, y):
    numerator = 0
    denominator = 0 
    for i in range(0, len(x)):
        if x[i] == y[i] == 1:
            numerator+=1
            denominator+=1
        elif (str(x[i])=='nan' and y[i]==1) or (x[i]==1 and str(y[i])=='nan'):
            denominator+=1
    if denominator !=0:
        return numerator/denominator 
    else:
        return 'NA'
                
for i in range(0, len(data.columns)):
    for j in range(i, len(data.columns)):
        similarity_df.at[data.columns[i], data.columns[j]] = jaccard_similarity(data[data.columns[i]], data[data.columns[j]])
      
# taking data to test 
data_test_1 = data.iloc[14000:, 0:75]
data_test_2 = data.iloc[14000:, 75:]

# checking the top 3 items for evaluation
def check_the_closest_n_items(item_id,n):
    dic_sim = {}
    for cols in similarity_df.columns:
        sim = max(similarity_df[item_id][cols],similarity_df[cols][item_id])
        dic_sim[cols] = sim
    dic_items = dict(sorted(dic_sim.items(), key=lambda item: item[1], reverse=True))
    j = 0
    dic_top = {}
    for i in dic_items:
        if j<n+1:
            if j!=0:
                dic_top[i] = dic_items[i]
            j = j+1
    return dic_top

check_the_closest_n_items('199811071',5)

# get the prediction from the different users based on their previous supscription
def get_scores(user_id, item_id):
    numerator = 0.0
    denominator = 0.0
    similar_three_item = check_the_closest_n_items(item_id,3)        
    for i in similar_three_item:
        if i in data_test_1.columns:
            if str(data_test_1[i][user_id]) != 'nan':
                numerator += data_test_1[i][user_id]*similar_three_item[i]
                denominator += similar_three_item[i]
    if denominator == 0.0:
        return 0.0
    else:
        return numerator/denominator         
        
#get_scores(14109,'199811071')

# making prediction on the test set
predictions = []
# test the similarity
for index, val in data_test_2.iterrows():
    for cols in data_test_2.columns:
        if data_test_2.at[index,cols] == 1:
            # check the measure of the score
            predictions.append([1, get_scores(14000,cols)])
        else:
            predictions.append([0, get_scores(14000,cols)])
            
prediction_arr = pd.DataFrame(predictions, columns = ["target","predicted"]) 

# accuracy score
accuracy_score(prediction_arr["target"], prediction_arr["predicted"])
   
# classification report    
target_names = ['class 0', 'class 1']
print(classification_report(prediction_arr["target"], prediction_arr["predicted"], target_names=target_names))

# confusion metrics
print("Confusion metrics: COllaborative filtering")
confusion_matrix(prediction_arr["target"], prediction_arr["predicted"])

#############################################################################################################################
# CONTENT BASED FILTERING
#############################################################################################################################

games_and_genre = pd.read_csv("data/games_and_genre_2.csv", encoding='latin1')
streamer_details = pd.read_csv("data/streamer_details.csv")
streamer_games = pd.read_csv("data/streamers_games.csv")

streamer_games["formatted"] = ''

streamer = streamer_games["games"][0].split(",")

def get_games_comma_seperated(streamer):
    streamer = streamer.split(",")
    final_str = []
    for i in streamer:
        i = re.sub(r"[^a-zA-Z0-9]","",i).lower()
        final_str.append(i.strip())
    return final_str

streamer_games["formatted"] = streamer_games["games"].apply(lambda x: get_games_comma_seperated(x))


# formatting the names of the games
games_and_genre["game_2"] = games_and_genre["game"].apply(lambda x: re.sub(r"[^a-zA-Z0-9]","",x).lower())

# getting the gamers details
streamer_games_types = streamer_details[['id','login']]
streamer_games_types = pd.merge(streamer_games_types, streamer_games, how='left', left_on="login", right_on="streamer")


# get the stopwords object
stops = set(stopwords.words('english'))

# getting the lemmatizer object 
lemmatizer = WordNetLemmatizer()
  
# function preprocess the text
def preprocessing_text(arr):
    mod_arr = []
    for desc in arr:
        #remove numbers
        desc = re.sub(r'[0-9]', " ", desc)
        # remove punctuations
        desc = re.sub(r'[^\w\s]'," ", desc)
        # remove stopwords
        string_final = []
        for i in desc.split(" "):
            if str(i).strip() != '' and str(i).strip() not in stops:
                string_final.append(lemmatizer.lemmatize(str(i).strip()))
        # convert list back to string
        string_final = ' '.join(string_final).strip() 
        # appending both the strings
        mod_arr.append(string_final)        
    return mod_arr        

# getting all the details of the gamer
def get_streamer_games_details(games):
    # variables to store the different properties of games
    modes = []
    genres = []
    description = []
    # going through every game
    for game in games:
        try:
            # get modes
            get_modes = str(games_and_genre.query("game_2 == '"+str(game)+"'")["Modes"][games_and_genre.query("game_2 == '"+str(game)+"'")["Modes"].index[0]]).replace(u'\xa0',u'')
            for mode in get_modes.split(","):
                if mode.lower().strip() != 'nan':
                    modes.append(mode.lower().strip())
            modes = list(set(modes))
            # get Genre
            get_genre = str(games_and_genre.query("game_2 == '"+str(game)+"'")["Genre"][games_and_genre.query("game_2 == '"+str(game)+"'")["Genre"].index[0]]).replace(u'\xa0',u'')
            for genre in get_genre.split(","):
                if genre.lower().strip() != 'nan':
                    genres.append(genre.lower().strip())
            genres = list(set(genres))
            # get games description
            get_description = str(games_and_genre.query("game_2 == '"+str(game)+"'")["Description"][games_and_genre.query("game_2 == '"+str(game)+"'")["Description"].index[0]]).replace(u'\xa0',u'')
            if get_description.lower().strip() != 'nan':
                description.append(get_description.lower().strip())
            description = list(set(description))
            description = preprocessing_text(description)
        except:
            pass
    #returning values
    return modes+genres+description

streamer_games_types["gaming_details"] = streamer_games_types["formatted"].apply(lambda x: get_streamer_games_details(x))

# getting all the meta data as a single string        
streamer_games_types["gaming_details_string"] = streamer_games_types["gaming_details"].apply(lambda x: ' '.join(x).strip())

# getting all the details of the games
streamer_games_types["game_and_gaming_details_string"] = streamer_games_types.apply(lambda x: ' '.join(x[4]+x[5]).strip(), axis=1)

# conversting meta data from the games description
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(streamer_games_types["gaming_details_string"])

# checking the cosine similarit between all the vectors based on the meta datainformation
from sklearn.metrics.pairwise import cosine_similarity

# building item-item collaborative filtering
similarity_df_metadata = pd.DataFrame(np.zeros((99,99), dtype=float), columns=data.columns)

# updating the indexes 
similarity_df_metadata.index = data.columns

# creating the similarity metrics among different gamers based on the similarities between them
for i in range(0,99):
    for j in range(i, 99):
        similarity_df_metadata.at[str(streamer_games_types["id"][13]), str(streamer_games_types["id"][j])] = cosine_similarity(X[j],X[i])[0][0]

def check_the_closest_n_items_metadata(item_id,n):
    dic_sim = {}
    for cols in similarity_df_metadata.columns:
        sim = max(similarity_df_metadata[item_id][cols],similarity_df_metadata[cols][item_id])
        dic_sim[cols] = sim
    dic_items = dict(sorted(dic_sim.items(), key=lambda item: item[1], reverse=True))
    j = 0
    dic_top = {}
    for i in dic_items:
        if j<n+1:
            if j!=0:
                dic_top[i] = dic_items[i]
            j = j+1
    return dic_top

check_the_closest_n_items_metadata('43691',5)

def get_scores_metadata(user_id, item_id):
    numerator = 0.0
    denominator = 0.0
    similar_three_item = check_the_closest_n_items_metadata(item_id,3)        
    for i in similar_three_item:
        if i in data_test_1.columns:
            if str(data_test_1[i][int(user_id)]) != 'nan':
                numerator += data_test_1[i][int(user_id)]*similar_three_item[i]
                denominator += similar_three_item[i]
    if denominator == 0.0:
        return 0.0
    else:
        return numerator/denominator         

get_scores_metadata('14005','639654714')
    

# making prediction on the test set
predictions_meta = []
# test the similarity
for index, val in data_test_2.iterrows():
    for cols in data_test_2.columns:
        if data_test_2.at[index,cols] == 1:
            # check the measure of the score
            predictions_meta.append([1, get_scores_metadata(14000,cols)])
        else:
            predictions_meta.append([0, get_scores_metadata(14000,cols)])
            
prediction_arr_meta = pd.DataFrame(predictions_meta, columns = ["target","predicted"]) 

# accuracy score
accuracy_score(prediction_arr_meta["target"], prediction_arr_meta["predicted"])

target_names = ['class 0', 'class 1']
print(classification_report(prediction_arr_meta["target"], prediction_arr_meta["predicted"], target_names=target_names))


print("Confusion metrics: Content based filtering")
confusion_matrix(prediction_arr_meta["target"], prediction_arr_meta["predicted"])


#############################################################################################################################
# STACKING THE TWO MODELS
#############################################################################################################################
    
data_test_1_old = data.iloc[0:14000, 75:]
#data_test_2_old = data.iloc[0:14000:, 75:]

# transforming the training dataset to implement the stacked model
def stacked_mode_results(user_id, item_id, n):
    print(user_id," -> ", item_id)
    arr = []
    five_similar = check_the_closest_n_items(item_id, n)
    five_meta_data = check_the_closest_n_items_metadata(item_id,n)
    arr.append(str(item_id))
    arr.append(str(user_id))
    i = 0
    for j in five_similar:
        arr.append(five_similar[j])
        i+=1
    if i == 0:
        arr.append(0.0)
        arr.append(0.0)
        arr.append(0.0)
    elif i == 1:
        arr.append(0.0)
        arr.append(0.0)
    elif i == 2:
        arr.append(0.0)
    
    i = 0
    for j in five_meta_data:
        arr.append(five_meta_data[j])
        i+=1
    if i == 0:
        arr.append(0.0)
        arr.append(0.0)
        arr.append(0.0)
    elif i == 1:
        arr.append(0.0)
        arr.append(0.0)
    elif i == 2:
        arr.append(0.0)
    if str(data_test_1_old[item_id][int(user_id)]) != 'nan':
        arr.append(data_test_1_old[item_id][int(user_id)])
    else:
        arr.append(0)
    return arr 

# dataset to stack the two models
stacked_dataset_train = []
# test the similarity
for index, val in data_test_1_old.iterrows():
    for cols in data_test_1_old.columns:
        stacked_dataset_train.append(stacked_mode_results(index, cols, 3))

# creating the dataframe from the matrix
stacked_dataset_train_df = pd.DataFrame(stacked_dataset_train, columns=['streamer','user_id','prob_1','prob_2','prob_3','prob_4','prob_5','prob_6','outcome'])

# seperaing the features and the output
X_training = stacked_dataset_train_df[['streamer','prob_1','prob_2','prob_3','prob_4','prob_5','prob_6']] #,'prob_4','prob_5','prob_6'
y_training = stacked_dataset_train_df['outcome']

# using SMOTE to oversample the dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_training, y_training)

# building the classifier
clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(X_res, y_res)

# decision tree
clf = DecisionTreeClassifier(random_state=0, max_depth=2)
clf.fit(X_res, y_res)

# creating the dataset to for testing

#data_test_2 = data.iloc[14000:, 75:]
def stacked_mode_results(user_id, item_id, n):
    print(user_id," -> ", item_id)
    arr = []
    five_similar = check_the_closest_n_items(item_id, n)
    five_meta_data = check_the_closest_n_items_metadata(item_id,n)
    arr.append(str(item_id))
    arr.append(str(user_id))
    i = 0
    for j in five_similar:
        arr.append(five_similar[j])
        i+=1
    if i == 0:
        arr.append(0.0)
        arr.append(0.0)
        arr.append(0.0)
    elif i == 1:
        arr.append(0.0)
        arr.append(0.0)
    elif i == 2:
        arr.append(0.0)
    
    i = 0
    for j in five_meta_data:
        arr.append(five_meta_data[j])
        i+=1
    if i == 0:
        arr.append(0.0)
        arr.append(0.0)
        arr.append(0.0)
    elif i == 1:
        arr.append(0.0)
        arr.append(0.0)
    elif i == 2:
        arr.append(0.0)
    if str(data_test_2[item_id][int(user_id)]) != 'nan':
        arr.append(data_test_2[item_id][int(user_id)])
    else:
        arr.append(0)
    return arr 

# dataset to stack the two models
stacked_dataset = []
# test the similarity
for index, val in data_test_2.iterrows():
    for cols in data_test_2.columns:
        stacked_dataset.append(stacked_mode_results(index, cols, 3))

stacked_dataset_test_df = pd.DataFrame(stacked_dataset, columns=['streamer','user_id','prob_1','prob_2','prob_3','prob_4','prob_5','prob_6','outcome'])

X_testing = stacked_dataset_test_df[['streamer','prob_1','prob_2','prob_3','prob_4','prob_5','prob_6']] #,'prob_4','prob_5','prob_6' 
y_testing = stacked_dataset_test_df['outcome']

y_pred_testing = clf.predict(X_testing)

# accuracy score
print(accuracy_score(y_pred_testing, y_testing))
# classification report
target_names = ['class 0', 'class 1']
print(classification_report(y_testing, y_pred_testing, target_names=target_names))

print("Confusion metrics: Content based filtering")
confusion_matrix(y_testing, y_pred_testing)
