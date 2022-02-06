import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random import seaborn as sns
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud data= pd.read_csv('../3D Objects/movies.csv')
data.shape data.head() data.info()
movies = data['movieId'].unique().tolist() len(movies)
ratings_data=pd.read_csv('../3D Objects/ratings.csv',sep=',') ratings_data.shape
ratings_data.head() ratings_data['rating'].min() ratings_data['rating'].max() data.shape data.isnull().any() ratings_data.shape ratings_data.isnull().any()
 
tags_data=pd.read_csv('../3D Objects/tags.csv',sep=',') tags_data.shape
tags_data.isnull().any() tags_data.head() tags_data=tags_data.dropna() tags_data.isnull().any()
unique_tags=tags_data['tag'].unique().tolist() len(unique_tags) drama_movies=data['genres'].str.contains('Drama') data[drama_movies].head()
drama_movies.shape
comedy_movies = data['genres'].str.contains('Comedy') data[comedy_movies].head()
comedy_movies.shape
tag_search = tags_data['tag'].str.contains('dark') tags_data[tag_search].head() ratings_data.head()
del ratings_data['timestamp'] ratings_data.head()
#merging two dataframes "movies.csv" and "ratings.csv" movie_data_ratings_data=data.merge(ratings_data,on = 'movieId',how = 'inner') movie_data_ratings_data.head(3)
data=movie_data_ratings_data
 
movie_data_ratings_data.groupby('title')['rating'].mean().sort_values(ascending=Fa lse).head()
movie_data_ratings_data.groupby('title')['rating'].count().sort_values(ascending=Fa lse).head()
ratings = pd.DataFrame(movie_data_ratings_data.groupby('title')['rating'].mean())


ratings['num of ratings'] = pd.DataFrame(movie_data_ratings_data.groupby('title')['rating'].count())


ratings.head() sns.set_style('white')
%matplotlib inline plt.figure(figsize =(10, 4))

ratings['num of ratings'].hist(bins = 70) # plot graph of 'num of ratings column' plt.figure(figsize =(10, 4))

ratings['rating'].hist(bins = 70) # displays low rated movies
low_rated = movie_data_ratings_data['rating']<4.0 movie_data_ratings_data[low_rated].head()
#total number of unique movie genre unique_genre=data['genres'].unique().tolist()
 
len(unique_genre)
#top 25 most rated movies most_rated =
movie_data_ratings_data.groupby('title').size().sort_values(ascending=False)[:25] most_rated.head(25)
#slicing out columns to display only title and genres columns from movies.csv data[['title','genres']].head()
# here we extract year from title
data['year'] =data['title'].str.extract('.*\((.*)\).*',expand = False) data.head(5)
#define a function that counts the number of times each genre appear: def count_word(df, ref_col, liste):
keyword_count = dict()
for s in liste: keyword_count[s] = 0
for liste_keywords in df[ref_col].str.split('|'):
if type(liste_keywords) == float and pd.isnull(liste_keywords): continue for s in liste_keywords:
if pd.notnull(s): keyword_count[s] += 1
# convert the dictionary in a list to sort the keywords by frequency keyword_occurences = []
for k,v in keyword_count.items(): keyword_occurences.append([k,v])
keyword_occurences.sort(key = lambda x:x[1], reverse = True)
 
return keyword_occurences, keyword_count #here we make census of the genres: genre_labels = set()
for s in data['genres'].str.split('|').values: genre_labels = genre_labels.union(set(s))
#counting how many times each of genres occur: keyword_occurences, dum = count_word(data, 'genres', genre_labels) keyword_occurences
# Function that control the color of the words
def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
h = int(360.0 * tone / 255.0) s = int(100.0 * 255.0 / 255.0)
l = int(100.0 * float(random_state.randint(70, 120)) / 255.0) return "hsl({}, {}%, {}%)".format(h, s, l)



#Finally, the result is shown as a wordcloud: words = dict()
trunc_occurences = keyword_occurences[0:50] for s in trunc_occurences:
words[s[0]] = s[1]
tone = 100 # define the color of the words
 
f, ax = plt.subplots(figsize=(14, 6))
wordcloud = WordCloud(width=550,height=300, background_color='black', max_words=1628,relative_scaling=0.7,
color_func = random_color_func, normalize_plurals=False)
wordcloud.generate_from_frequencies(words) plt.imshow(wordcloud, interpolation="bilinear") plt.axis('off')
plt.show()
# lets display the same result in the histogram fig = plt.figure(1, figsize=(18,13))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)] x_label = [i[0] for i in trunc_occurences] plt.xticks(rotation=85, fontsize = 15) plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0) ax2.bar(x_axis, y_axis, align = 'center', color='r')
plt.title("Popularity of Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)
plt.show()
 
moviesonly=np.array(movie_data_ratings_data['title'])[:10000] print(moviesonly)
discrete_movies = []
for movie in moviesonly:
if movie not in discrete_movies: discrete_movies.append(movie)
print(discrete_movies) movie_list =[]
for movie in moviesonly: movie_list.append(discrete_movies.index(movie))
print(movie_list) matrix=np.array([movie_list,movie_data_ratings_data['rating'][1:]]) print(matrix)
