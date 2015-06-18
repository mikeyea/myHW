##Homework 5##
##Mike Yea##
```python
import pandas as pd
import matplotlib.pyplot as plt
# read in 'imdb_1000.csv' and store it in a DataFrame named movies
movies = pd.read_table('imdb_1000.csv', sep=',')
# check the number of rows and columns
movies.shape
# check the data type of each column
movies.dtypes
# calculate the average movie duration
movies.duration.mean() #121 minutes
# sort the DataFrame by duration to find the shortest and longest movies
movies.sort('duration') #Freaks is shortest.  Hamlet is longest
# create a histogram of duration, choosing an "appropriate" number of bins
movies.duration.plot(kind='hist', bins=20)
# use a box plot to display that same data
movies.duration.plot(kind='box')
'''
INTERMEDIATE LEVEL
'''

# count how many movies have each of the content ratings
movies.content_rating.value_counts()
#'R':460; 'PG-13':189; 'PG': 123; 'NOT RATED': 65; 'APPROVED': 47; 'UNRATED': 38
#'G': 32; 'PASSED': 7; 'NC-17': 7
# use a visualization to display that same data, including a title and x and y labels
movies.content_rating.value_counts().plot(kind='bar', title='Number of Movies by Content Rating')
plt.xlabel('Rating')
plt.ylabel('Number of Movies')
# convert the following content ratings to "UNRATED": NOT RATED, APPROVED, PASSED, GP
movies[movies.content_rating.isin(['NOT RATED', 'APPROVED', 'PASSED', 'GP'])] #check out the dataframe first
movies.content_rating.replace(['NOT RATED', 'APPROVED', 'PASSED', 'GP'], 'UNRATED', inplace=True)
# convert the following content ratings to "NC-17": X, TV-MA
movies[movies.content_rating.isin(['X', 'TV-MA'])]
movies.content_rating.replace(['X', 'TV-MA'], 'NC-17', inplace=True)
# count the number of missing values in each column
movies.isnull().sum()
#content_rating has 3 missing values
# if there are missing values: examine them, then fill them in with "reasonable" values
movies.content_rating.fillna(value='NA', inplace=True)
# calculate the average star rating for movies 2 hours or longer,
# and compare that with the average star rating for movies shorter than 2 hours
movies[(movies.duration > 120) | (movies.duration == 120)].star_rating.mean()
movies[movies.duration < 120].star_rating.mean()
#movies >= 120 rating is 7.95; movies < 120, rating is 7.84
# calculate the average duration for each genre
movies.groupby('genre').duration.mean()
#too many to type in intelligibly
# use a visualization to detect whether there is a relationship between star rating and duration
movies.plot(kind='scatter', x='duration', y='star_rating')

'''
ADVANCED LEVEL
'''
# visualize the relationship between content rating and duration
movies['content_rating_num'] = movies.content_rating.map({'G':0, 'PG':1, 'PG-13': 2, 'R': 3, 'NC-17': 4, 'UNRATED': 5})
movies.plot(kind='scatter', x='content_rating_num', y='duration')
#content rating needs to be converted to a number
# determine the top rated movie (by star rating) for each genre
movies.sort(['genre','star_rating']) #not sure why this is not working
# check if there are multiple movies with the same title, and if so, determine if they are the same movie
movies.title.duplicated().value_counts()
#There are 4 rows that returned true
movies[movies.duplicated()]
#returns empty dataframe.  They must not be duplicates        
# calculate the average star rating for each genre, but only include genres with at least 10 movies
movies.genre.value_counts(dropna=False)
#pseudo: create a dataframe that has the original dataframe grouped by genre and has count and mean, and then
# filter for count > 9
genre_counts = movies.groupby('genre').star_rating.agg(['count', 'mean'])
genre_counts[genre_counts['count'] > 9] 
'''
```