'''
#Recommendation System
#Content-Based Filtering
# Here I am woriking with the MovieLens Dataset. 
# This dataset contains user generated movie ratings from the website MovieLens (https://movielens.org/).
#Lets load the movies file
'''
import numpy as np 
import pandas as pd 
movies_df = pd.read_csv("movies.csv")
print(movies_df.head())
'''
#In order to be able to work with the movie_genre column, we need to transform it to what is called "dummy variables".
#This is a way to convert a categorical variable (e.g. Animation, Comedy, Romance...), into multiple columns (one column named Action, one named Comedy, etc).
#For each movie, these dummy columns will have a value of 0 except for those genres the movie has.
# we convert the movie genres to a set of dummy variables 
'''
movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  
print(movies_df.head() ) 
# print(movies_df.columns)
movies_df.drop('(no genres listed)', axis=1, inplace=True)
# print(movies_df.columns)

'''
#So for example, the movie with an id of 1 Toy Story, belongs to the genres Animation, Children's and Comedy, and thus the columns Animation, 
#Children's and Comedy have a value of 1.
'''

movie_categories = movies_df.columns[3:]  
print(movies_df.loc[0]) 

'''
#Content filtering is a simple way to build a recommendation system. Here, items (in this problem movies) are mapped to a set of features (genres).
#To recommend a user an item, first that user has to provide his/her preferences regarding those features.
#So in this example, the user has to tell the system how much does he or she like each movie genre.
#Right now we have all the movies mapped into genres. We just need to create a user and map that user into those genres.
#Let's create a user with strong preference for action, adventure and fiction movies.
'''

from collections import OrderedDict
user_preferences = OrderedDict(zip(movie_categories, []))
user_preferences['Action'] = 5  
user_preferences['Adventure'] = 5  
user_preferences['Animation'] = 1  
user_preferences["Children's"] = 1  
user_preferences["Comedy"] = 3  
user_preferences['Crime'] = 2  
user_preferences['Documentary'] = 1  
user_preferences['Drama'] = 1  
user_preferences['Fantasy'] = 5  
user_preferences['Film-Noir'] = 1  
user_preferences['Horror'] = 2  
user_preferences['Musical'] = 1  
user_preferences['Mystery'] = 3  
user_preferences['Romance'] = 1  
user_preferences['Sci-Fi'] = 5  
user_preferences['War'] = 3  
user_preferences['Thriller'] = 2  
user_preferences['Western'] =1  

'''
# Once we have users with their movie genre preferences and the movies mapped into genres, 
# to compute the score of a movie for a specific user, we just need to calculate the dot product of that movie genre vector 
# with that user preferences vector.
'''


def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])
    #return (np.dot(vector_1,vector_2))

def get_movie_score(movie_features, user_preferences):  
    return dot_product(movie_features, user_preferences)

'''
#compute the score of the movie 'Toy Story' (a children's animation movie) for the sample user.
'''

toy_story_features = movies_df.loc[0][movie_categories]  
print(toy_story_features )

'''
#compute the predicted score by taking toy_story_features and user_preferences
'''

toy_story_user_predicted_score = dot_product(toy_story_features, user_preferences.values())  
print(toy_story_user_predicted_score)  

'''
# So for the user, Toy Story, has a score of 15. Which does not mean much by itself, 
# but helps us comparing how good of a recommendation Toy Story is compared to other movies.
'''

'''
#calculate the score for Die Hard (a thrilling action movie):
'''
print(movies_df[movies_df.title.str.contains('Die Hard')]  )


die_hard_id = 1036  
die_hard_features = movies_df[movies_df.movieId==die_hard_id][movie_categories]  
print(die_hard_features.T  )

'''
:Note
#1017 is the dataframe row index for Die Hard, not the movie index in the movielens dataset
'''

die_hard_user_predicted_score = dot_product(die_hard_features.values[0], user_preferences.values())  
print(die_hard_user_predicted_score  )

'''
So we see that Die Hard gets an score of 9 vs a 15 for Toy Story. So Toy Story would be recommended before Die Hard. 
'''

'''
Function to give the best 10 recommendations
''' 

def get_movie_recommendations(user_preferences, n_recommendations):  
    #we add a column to the movies_df dataset with the calculated score for each movie for the given user
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score, 
                                                           args=([user_preferences.values()]), axis=1)
    return movies_df.sort_values(by=['score'], ascending=False)['title'][:n_recommendations]

print(get_movie_recommendations(user_preferences, 10)  )