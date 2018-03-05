import pandas as pd
import numpy as np

from pyzipcode import ZipCodeDatabase
zcdb = ZipCodeDatabase()


import colorsys

def get_N_HexCol(N=5, a=0.5, b=0.5):

    HSV_tuples = [(x*1.0/N, a, b) for x in xrange(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb))
        hex_out.append("#" + "".join(map(lambda x: chr(x).encode('hex'),rgb)))
    return hex_out

def get_zip_data(zipcode, zip_feature):
    """
    
    Given a zip code, retrieve a geographical feature such as 
    State, Latitude, Longitude, etc
    
    """
    try:
        return getattr(zcdb[zipcode], zip_feature)
    except IndexError:
        return np.nan
    
def get_user_features(users, items, ratings):
    
    """
    
    Given a data set of users, movies, and movie ratings,
    generate user demographic and rating features
    
    """
    
    
    def get_user_demographics(users):
        
        """
        
        Generate demographic features from a set of users
        
        """
        
        state_dummies = pd.get_dummies(users['state'])
        sex_dummies = pd.get_dummies(users['sex'])
        occupation_dummies = pd.get_dummies(users['occupation'])
        user_demographics = pd.DataFrame()
        user_demographics['normed_age'] = users.age/130
        user_demographics = pd.concat([user_demographics, state_dummies, sex_dummies, occupation_dummies], axis = 1)
        user_demographics.index = users.user_id
        return user_demographics

    def get_user_ratings(items, ratings):
        
        """
        
        Generate rating features from a set of movies and user ratings
        
        """
        
        user_movie_ratings = pd.merge(items, ratings, how = 'right', left_on = 'movie id', right_on = 'movie_id')
        user_movie_ratings = user_movie_ratings.pivot(index = 'user_id', columns = 'movie_id', values = 'rating')
        user_movie_ratings.columns = user_movie_ratings.columns.astype(str)
        return user_movie_ratings



    if len(np.unique(users.user_id)) < len(users.user_id):
        raise ValueError('Error: Duplicate user IDs detected in "users" dataframe')
        
        
    if len(np.unique(items['movie id'])) < len(items['movie id']):
        raise ValueError('Error: Duplicate movie IDs detected in "items" dataframe')

       
    #Generate the demographic and rating features
    user_demographics = get_user_demographics(users)
    user_movie_ratings = get_user_ratings(items, ratings)
    
    #Normalize the rating features by the maximum rating
    max_rating = 5.
    user_movie_ratings = user_movie_ratings/max_rating
    
    #Combine demographic and rating featurs into one data frame
    user_features = pd.merge(user_demographics, user_movie_ratings,
                                   left_index = True, right_index = True, how = 'outer')
    
    #Set the data frame's index to the users' IDs
    user_features.index = users.user_id
    
    return user_features, user_demographics, user_movie_ratings



def get_item_features(items):
    
    """
    
    Extract the feature columns from the 'items' data frame
    
    """
    item_features = items.iloc[:,6:].copy()
    
    item_names = items['movie title'].copy()
    item_ids = items['movie id'].copy()
    
    item_features.index = item_ids
    
    return item_features, item_names, item_ids

def euclidean_score(vec1, vec2):
    
    """
    
    Compute a user similarity score based on
    euclidean distance of user feature vectors
    
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