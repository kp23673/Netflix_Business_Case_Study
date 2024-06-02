#!/usr/bin/env python
# coding: utf-8

# # Business Case: Netflix - Data Exploration and Visualisation

# # About NETFLIX
# 
# 

# Netflix is one of the most popular media and video streaming platforms. They have over 10000 movies or tv shows available on their platform, as of mid-2021, they have over 222M Subscribers globally. This tabular dataset consists of listings of all the movies and tv shows available on Netflix, along with details such as - cast, directors, ratings, release year, duration, etc.
# 
# 

# # Business Problem
# 
# 

# Analyze the data and generate insights that could help Netflix ijn deciding which type of shows/movies to produce and how they can grow the business in different countries.

# # Problem Statement:
# 

# Given the Netflix dataset, the problem statement could be to analyze the content available on Netflix and gain insights into its library. You may want to focus on understanding the distribution of content, popular genres, top-rated movies or TV shows, and trends over time. This can help Netflix make data-driven decisions for content acquisition and improve user experience.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("netflix.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.describe( include = 'all')


# In[6]:


df.info()


# In[7]:


df.nunique()


# In[8]:


df.isnull().sum()/ len(df) * 100


# In[9]:


# columns 
df.columns


# Since some columns have nested values, will unnest them and prepare final dataset¶
# 

# In[10]:


# Un-nesting Directors  columns: 

Dir_col = df['director'].apply(lambda x: str(x).split(", ")).tolist()
df_1 = pd.DataFrame(Dir_col, index = df['title'])
df_1 = df_1.stack()
df_1 = pd.DataFrame(df_1.reset_index())
df_1.rename(columns={0:'Directors'},inplace=True)
df_1 = df_1.drop(['level_1'],axis=1)
df_1.head(10)


# In[11]:


# Un-nesting cast  columns:

cast_col = df['cast'].apply(lambda x: str(x).split(", ")).tolist()
df_2 = pd.DataFrame(cast_col, index = df['title'])
df_2 = df_2.stack()
df_2 = pd.DataFrame(df_2.reset_index())
df_2.rename(columns={0:'Actors'},inplace=True)
df_2 = df_2.drop(['level_1'],axis=1)
df_2.head(10)


# In[12]:


# Un-nesting listed_in  columns:

lst_col = df['listed_in'].apply(lambda x: str(x).split(", ")).tolist()
df_3 = pd.DataFrame(lst_col, index = df['title'])
df_3 = df_3.stack()
df_3 = pd.DataFrame(df_3.reset_index())
df_3.rename(columns={0:'Genre'},inplace=True)
df_3 = df_3.drop(['level_1'],axis=1)
df_3.head(10)


# In[13]:


# Un-nesting country  columns:

country_col = df['country'].apply(lambda x: str(x).split(", ")).tolist()
df_4 = pd.DataFrame(country_col, index = df['title'])
df_4 = df_4.stack()
df_4 = pd.DataFrame(df_4.reset_index())
df_4.rename(columns={0:'Country'},inplace=True)
df_4 = df_4.drop(['level_1'],axis=1)
df_4.head(10)


# We are merging all un-nested dataframes using merge function 

# In[14]:


df_5 = df_2.merge(df_1,on=['title'],how='inner')

df_6 = df_5.merge(df_3,on=['title'],how='inner')

df_7 = df_6.merge(df_4,on=['title'],how='inner')

df_7.head()


# In[15]:


df_7.shape


# Merging unnested data with the given dataframe
# 
# 

# In[16]:


# merging unnested data with the given dataframe

df = df_7.merge(df[['show_id', 'type', 'title', 'date_added',
       'release_year', 'rating', 'duration']],on=['title'],how='left')
df.head()


# In[17]:


df.shape


# In[18]:


df.isnull().sum()


# There were some missing values will treat them
# 
# 

# In[19]:


Total_null = df.isnull().sum().sort_values(ascending = False)
Percentage = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", df.shape[0])

missing_data = pd.concat([Total_null,Percentage.round(2)],axis=1,keys=['Total Missing','In Percentage'])
missing_data.head(10)


# Above table gives missing values summary in absolute value and in Percentage, date added has the maximum missing values

# Missing value treatment¶
# 

# In[20]:


# some columns having nan which is missing value, we have to replace

df['Actors'].replace(['nan'],['Unknown Actor'],inplace=True)
df['Directors'].replace(['nan'],['Unknown Director'],inplace=True)
df['Country'].replace(['nan'],[np.nan],inplace=True)
df.head()


# In[21]:


Total_null = df.isnull().sum().sort_values(ascending = False)
Percentage = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
print("Total records = ", df.shape[0])

missing_data = pd.concat([Total_null,Percentage.round(2)],axis=1,keys=['Total Missing','In Percentage'])
missing_data.head(10)


# After replacing string nan with np.nan, actual null values of country went upto 5.89 %
# 
# 

# In[22]:


df[df.duration .isnull()]


# duration and rating columns got messed up and values got exchanged will add rating column values into duration column missing values

# In[23]:


df.loc[df['duration'].isnull(),'duration'] = df.loc[df['duration'].isnull(),'duration'].fillna(df['rating'])
df.loc[df['rating'].str.contains('min', na=False),'rating'] = 'NR'
df['rating'].fillna('NR',inplace=True)
df.isnull().sum()


# Filling missing values of date added column with mode value with respective release years
# 
# 

# In[24]:


for i in df[df['date_added'].isnull()]['release_year'].unique():
    date = df[df['release_year'] == i]['date_added'].mode().values[0]
    df.loc[df['release_year'] == i,'date_added'] = df.loc[df['release_year']==i,'date_added'].fillna(date)


# In[25]:


df[df.Country.isna()]


# Filling missing values of country column with mode value with respective directors
# 
# 

# In[26]:


for i in df[df['Country'].isnull()]['Directors'].unique():
    if i in df[~df['Country'].isnull()]['Directors'].unique():
        country = df[df['Directors'] == i]['Country'].mode().values[0]
        df.loc[df['Directors'] == i,'Country'] = df.loc[df['Directors'] == i,'Country'].fillna(country)


# In[27]:


df.isnull().sum()


# In[28]:


for i in df[df['Country'].isnull()]['Actors'].unique():
    if i in df[~df['Country'].isnull()]['Actors'].unique():
        imp = df[df['Actors'] == i]['Country'].mode().values[0]
        df.loc[df['Actors']==i, 'Country'] = df.loc[df['Actors'] == i, 'Country'].fillna(imp)


# In[29]:


df['Country'].fillna('Unknown Country',inplace=True)       
df.isnull().sum()


# Remaining missing values will be replaced using actors column
# 
# 

# Now missing values handling is over, will deep dive into data analysis

# # 1. How has the number of movies released per year changed over the last 20-30 years?

# In[30]:


df['date_added'] = pd.to_datetime(df['date_added'])
df['release_year'] = df['date_added'].dt.year
movies_df = df[df['type'] == 'Movie']
movies_per_year = movies_df['release_year'].value_counts().sort_index()


# In[31]:


plt.figure(figsize=(10, 6))
plt.plot(movies_per_year.index, movies_per_year.values, color='b')
plt.xlabel('Year')
plt.ylabel('Number of Movies Released')
plt.title('Number of Movies Released per Year (Last 20-30 Years)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# According to line chart, from year 2014 there is rise in movies till 2019 but from 2019 there fall of movies.

# # 2. Comparison of tv shows vs. movies.
# 

# In[32]:


# Filter the DataFrame for movies and TV shows

movies_df = df[df['type'] == 'Movie']
tv_shows_df = df[df['type'] == 'TV Show']
    
    


# In[33]:


# Find the number of movies produced in each country and pick the top 10 countries

top_movies_countries = movies_df['Country'].value_counts().head(10)


# In[34]:


# Find the number of TV shows produced in each country and pick the top 10 countries

top_tv_shows_countries = tv_shows_df['Country'].value_counts().head(10)


# In[35]:


# Plot the visualization: 

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
top_movies_countries.plot(kind='bar', color='b')
plt.xlabel('Country')
plt.ylabel('Number of Movies')
plt.title('Top 10 Countries with the Most Movies')

plt.subplot(1, 2, 2)
top_tv_shows_countries.plot(kind='bar', color='g')
plt.xlabel('Country')
plt.ylabel('Number of TV Shows')
plt.title('Top 10 Countries with the Most TV Shows')

plt.tight_layout()
plt.show()


# According to the both the graphs, The United States is the country at the top for producing Movies compared to TV shows.
# Also, India is second for producing Movies.
# Japan and The United Kingdom are the top countries to produced the TV shows. 

# # 3. What is the best time to launch a TV show?

# In[36]:


# Find which is the best month to release the Tv-show or the movie. Do the analysis separately for Tv-shows and Movies


# In[37]:


# Convert the "date_added" column to a datetime data type:

df['date_added'] = pd.to_datetime(df['date_added'])


# In[38]:


# Separate the DataFrame into TV shows and movies:

tv_shows_df = df[df['type'] == 'TV Show']
movies_df = df[df['type'] == 'Movie']


# In[39]:


# Group the data by month and calculate the count of TV shows and movies added in each month:

tv_shows_by_month = tv_shows_df['date_added'].dt.month.value_counts().sort_index()
movies_by_month = movies_df['date_added'].dt.month.value_counts().sort_index()


# In[44]:


# Plot the results:

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=tv_shows_by_month.index, y=tv_shows_by_month.values, data = df)
plt.xlabel('Month')
plt.ylabel('Number of TV Shows Released')
plt.title('Number of TV Shows Released by Month')

plt.subplot(1, 2, 2)
sns.boxplot(x=movies_by_month.index, y=movies_by_month.values, data = df)
plt.xlabel('Month')
plt.ylabel('Number of Movies Released')
plt.title('Number of Movies Released by Month')

plt.tight_layout()
plt.show()


# To launch TV shows month of December, July and August is best period.
# To launch Movie month of July, January and October is best period.

# # 4. Analysis of actors of different types of movies.

# In[41]:


# Identify the top 10 actors who have appeared in most movies.

# Filter out the rows where the "actors" is not unknown:

df = df[df['Actors'] != 'Unknown Actor']

# Group the data by actor and count the number of unique titles:

actors_counts = df.groupby('Actors')['title'].nunique().sort_values(ascending=False)
top_10_actors = actors_counts.head(10)
# plot
plt.figure(figsize=(10, 6))
sns.barplot(x = top_10_actors.values, y = top_10_actors.index, data = df)
plt.xlabel('Number of Titles')
plt.ylabel('Actors')
plt.title('Top 10 Actors with Most Titles (Movies)')
plt.tight_layout()
plt.show()


# Anupam Kher, Shah Rukh Khan, Naseeruddin Shah are most popular actors who appeared in the most movies.
# There is list of top 10 actors who are appeared most in the Bollywood movies.

# # 5. Which genre movies are more popular or produced more

# # Univariate Analysis

# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_movies = df[df['type']=='Movie']

df_genre = df_movies.groupby(['Genre']).agg({"title":"nunique"}).reset_index().sort_values(by=['title'],ascending=False)[:10]
plt.figure(figsize = (15,6))
sns.barplot(y = "Genre",x = 'title', data = df_genre)
plt.xticks(rotation = 60)
plt.title('Top 10 Genres')
plt.show()






# International Movies, Dramas, Comedies are most popular genre as well as it more focused in recent years.

# # 6. Understanding what content is available in different countries

# In[43]:


import pandas as pd
import matplotlib.pyplot as plt


# Group the data by content title and count the number of unique countries for each content
content_countries_count = df.groupby('title')['Country'].nunique().reset_index()

# Sort the DataFrame in descending order based on the number of countries
top_10_content_countries = content_countries_count.sort_values(by='Country', ascending=False).head(10)

# Create the horizontal bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Country', y='title', data=top_10_content_countries, palette='viridis')
plt.xlabel('Number of Countries')
plt.ylabel('Content Title')
plt.title('Top 10 Content Available in Different Countries')
plt.tight_layout()
plt.show()




# From bar chart we can see the bar chart showing top 10 contents in different countries. 
# The professor and the Madman is on the top for the content.

# # Business Insights :
# 

# 
# A. Over the last 20-30 years, there is fluctuation in the movies released per year.
# 
# 
# B. The United States is the country at the top for producing Movies compared to TV shows. 
# 
# 
# C. India is second for producing Movies. 
# 
# 
# D. Japan and The United Kingdom are the top countries to produced the TV shows.
# 
# 
# E. To launch TV shows month of December, July and August is best period.
# 
# 
# F. To launch Movie month of July, January and October is best period.
# 
# 
# G. Anupam Kher, Shah Rukh Khan, Naseeruddin Shah are most popular actors who appeared in the most movies.
# 
# 
# H. International Movies, Dramas, Comedies are most popular genre as well as it more focused in recent years.
# 
# 
# I. The Professor and the Madman is on the top for understanding the content.
# 
# 
#     
#     

# # Recommendations:

# 
# A. Prevent the fluctuation of movies by good contents to movies. Add some good songs which can like by the old and new generation simultaneously. Give inspirational message to the public, so they can watch the movies at any cost.
# 
# 
# B. Compared to movies give priority to TV shows. To do this create a TV shows of real incidents that can be useful for new generation such as kids, students, etc. For the country which is having low rating in TV shows, give more ads on social media so people can take interest to watch TV shows.
# 
# 
# C. Add TV Shows/ movies in the month of February or May.
# 
# 
# D. While creating content, take into consideration the popular actors/directors for that country. Also take into account the director-actor combination which is highly recommended.
# 
# 
# E. Encourage the people to watch Documentaries and family movies/TV shows.
# 
# 
