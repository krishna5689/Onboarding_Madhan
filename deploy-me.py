import pip
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from pyzipcode import ZipCodeDatabase
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numba
import os

zcdb = ZipCodeDatabase()
def get_zip_data(zipcode, zip_feature):
    try:
        #return zcdb[zipcode].zip_feature
        return getattr(zcdb[zipcode], zip_feature)
    except IndexError:
        return np.nan

def get_user_features(users, items, ratings):
    
    
    def get_user_demographics(users):
        state_dummies = pd.get_dummies(users['state'])
        sex_dummies = pd.get_dummies(users['sex'])
        occupation_dummies = pd.get_dummies(users['occupation'])
        user_demographics = pd.DataFrame()
        user_demographics['normed_age'] = users.age/130
        user_demographics = pd.concat([user_demographics, state_dummies, sex_dummies, occupation_dummies], axis = 1)
        user_demographics.index = users.user_id
        return user_demographics

    def get_user_ratings(items, ratings):
        user_movie_ratings = pd.merge(items, ratings, how = 'right', left_on = 'movie id', right_on = 'movie_id')
        user_movie_ratings = user_movie_ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating')
        user_movie_ratings.columns = user_movie_ratings.columns.astype(str)
        return user_movie_ratings



    if len(np.unique(users.user_id)) < len(users.user_id):
        raise ValueError('Error: Duplicate user IDs detected in "users" dataframe')
        
        
    if len(np.unique(items['movie id'])) < len(items['movie id']):
        raise ValueError('Error: Duplicate movie IDs detected in "items" dataframe')

        
    user_demographics = get_user_demographics(users)
    user_movie_ratings = get_user_ratings(items, ratings)
    
    max_rating = 5.
    user_movie_ratings = user_movie_ratings/max_rating
    
    user_features = pd.merge(user_demographics, user_movie_ratings,
                                   left_index = True, right_index = True, how = 'outer')
    
    user_features.index = users.user_id
    
    return user_features, user_demographics, user_movie_ratings

def get_item_features(items):
    item_features = items.iloc[:,6:].copy()
    
    item_names = items['movie title'].copy()
    item_ids = items['movie id'].copy()
    
    item_features.index = item_ids
    
    return item_features, item_names, item_ids


def euclidean_score(vec1, vec2):
    """
    vec1, vec2 - numpy arrays
    
    """
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)

    
    mask = np.where((~np.isnan(vec1)) & (~np.isnan(vec2)))[0]
    

    weight = np.float64(len(mask))/len(vec1)
    
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    
    dist = pdist([vec1, vec2], "euclidean")[0]

    dist_score = 1./(1+dist)
    wdist_score = dist_score*weight
    
    return dist_score, wdist_score, weight


#####################################
# Load Data                         #
#####################################

#Load Data
data_dir = ''

#Read data
users = pd.read_csv(data_dir + 'u.user', sep = '|', names = ['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                    encoding = 'latin-1')

ratings = pd.read_csv(data_dir + 'u.data', sep = '\t', names = ['user_id', 'movie_id', 'rating', 'unix_timestamp'],
                    encoding = 'latin-1')

items = pd.read_csv(data_dir + 'u.item', sep = '|', 
                    names = ['movie id', 'movie title' ,'release date','video release date', 
                             'IMDb URL', 'unknown', 'Action', 'Adventure','Animation', 
                             'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                             'Thriller', 'War', 'Western'],
                    encoding = 'latin-1')


users['state'] = users.apply(lambda row: get_zip_data(row.zip_code, 'state'), axis = 1)


#########################################
# Create Features                       #
#########################################

user_features, user_demographics, normed_user_movie_ratings = get_user_features(users, items, ratings)
item_features, item_names, item_ids = get_item_features(items)


#########################################
# Compute Item Similarity               #
#########################################

itemF_sparse = sparse.csr_matrix(np.array(item_features))

item_similarity = pd.DataFrame(cosine_similarity(itemF_sparse))
item_similarity.index = item_ids.astype(str)
item_similarity.columns = item_ids.astype(str)


#########################################
# Model Function                        #
#########################################

###  Convert data structurs to Numpy  ###
#- Numpy supports faster indexing than pandas and many numpy functions
#- can be used with Numba for compilation

#--- User Features ---#
uf_arr = np.squeeze(user_features.values) #user feature values
uf_uid = np.squeeze(user_features.index.values) #user feature user IDs

#--- Movie Ratings ---#
umr_idx = np.squeeze(normed_user_movie_ratings.index.values) #movie rating user IDs
umr_cols = np.squeeze(normed_user_movie_ratings.columns.values.astype(int)) #movie IDs
umr_arr = np.squeeze(normed_user_movie_ratings.values) #user movie ratings (normed)

#--- User Demographic Features ---#
ud_idx = np.squeeze(user_demographics.index.values)
ud_arr = np.squeeze(user_demographics.values)

#--- Item Similarity ---#
its_idx = np.squeeze(item_similarity.index.values.astype(int)) #item similiarity movie ID
its_arr = np.squeeze(item_similarity.values) #item similarity value

@numba.jit
def predict_rating(user_id, numItems):
    
    #--- Generate features for the sample user, and all the remaining users ---#
    all_users_features = uf_arr[np.where(uf_uid != user_id)]
    
    user_idx = (user_id==umr_idx)
    sample_user_ratings = np.squeeze(umr_arr[user_idx])
    all_users_ratings = umr_arr[~user_idx]

    user_idx = (user_id==ud_idx)
    sample_user_demog = np.squeeze(ud_arr[user_idx])
    sample_user_features = np.append(sample_user_demog, sample_user_ratings, 0)
    #--- END ---#
    
    
    #--- Get Watched and Unwatched Items by Sample User, and Associated Movie IDs ---#
    nanidx = np.isnan(sample_user_ratings)
    watched_items = umr_cols[~nanidx]
    watched_ratings = sample_user_ratings[~nanidx]

    unwatched_items = umr_cols[nanidx]
    uwidx1 = np.in1d(umr_cols, unwatched_items) #unwatched movie index for user rating matrix
    uwidx2 = np.in1d(its_idx, unwatched_items) #unwatched movie index for item similarity matrix
    #--- END ---#
    
    #--- Compute sample user similarity to all other users ---#
    user_similarity = np.apply_along_axis(euclidean_score, 1, all_users_features, sample_user_features)[:,1]

    #--- Vectorized Loop - START ---#
    weights = user_similarity / np.max(user_similarity)
    uw_ratings = all_users_ratings[:,uwidx1]
    uw_ratings[np.isnan(uw_ratings)] = 0
    uw_score1 = np.dot(weights, uw_ratings) / np.nansum(weights)

    weights2 = its_arr[uwidx2,:]
    weights2 = weights2[:,~nanidx]
    weights2[np.isnan(weights2)] = 0
    watched_ratings[np.isnan(watched_ratings)] = 0
    uw_score2 = np.dot(weights2, watched_ratings) / np.nansum(weights2, axis = 1)
    #--- Vectorized Loop - END ---#
    
    rec_scores = (uw_score1 + uw_score2)/2. #score used to recommend items
    
    #--- Sort unwatched items by their rec_scores ---#
    nanscoreidx = np.isnan(rec_scores)
    rec_scores = rec_scores[~nanscoreidx]
    unwatched_items = unwatched_items[~nanscoreidx]
    sort_idx = np.argsort(rec_scores)[::-1]
    rec_scores = rec_scores[sort_idx]
    unwatched_items = unwatched_items[sort_idx]
    #--- END ---#
    
    if numItems <= 0:
        numItems = len(rec_scores)
         
    return rec_scores[0:numItems].tolist(), unwatched_items[0:numItems].tolist()



