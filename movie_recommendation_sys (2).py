#!/usr/bin/env python
# coding: utf-8

# # MOVIE RECOMMENDATION SYSTEM
# 
# The main goal of this machine learning project is to build a recommendation system that recommends movies to users A content-based recommender system is a type of recommender system that relies on the similarity between items to make recommendations. For example, if you’re looking for a new movie to watch, a content-based recommender system might recommend movies that are similar to ones you’ve watched in the past.
# 
# 

# In[3]:


import numpy as np
import pandas as pd  #importing libraries


# In[ ]:


credits=pd.read_csv('C:\\Users\\sriha\\OneDrive\\Desktop\\credits.csv')
movies=pd.read_csv('C:\\Users\\sriha\\OneDrive\\Desktop\\movies.csv')


# In[8]:


movies.head(1)


# # we have taken only useful features in these dataset

# In[9]:


movies=movies.merge(credits,on='title')     # here we have merged the two dataframes


# In[10]:


movies.head(1)


# In[11]:


movies.shape


# In[12]:


credits.shape


# In[13]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[14]:


movies.head()


# In[15]:


movies.isnull().sum()


# In[16]:


movies.dropna(inplace=True)


# In[17]:


movies.isnull().sum()


# In[18]:


movies.duplicated().sum()


# In[19]:


movies.iloc[0].genres


# In[20]:


import ast


# In[21]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L


# In[22]:


movies['genres']=movies['genres'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[25]:


movies['cast']


# In[26]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
             break
    return L 


# In[27]:


movies['cast'] = movies['cast'].apply(convert3)
movies.head()


# In[28]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# # up to here data preprocessing

# In[31]:


movies['overview'][0]


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[36]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[37]:


movies.head()


# In[38]:


new_df=movies[['movie_id','title','tags']]


# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[40]:


new_df.head()


# In[41]:


new_df['tags'][0]


# In[42]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[43]:


new_df.head()


# In[44]:


import nltk  #importing libraries


# In[45]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[46]:


def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)


# In[47]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[49]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[50]:


vector


# In[51]:


vector[0]


# In[52]:


cv.get_feature_names_out()


# In[53]:


ps.stem('danced')


# In[54]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[55]:


from sklearn.metrics.pairwise import cosine_similarity


# In[56]:


similarity = cosine_similarity(vector)


# In[57]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x: x[1])[1:6]


# In[58]:



def recommend (movie):
    
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate (similarity[index])), reverse=True, key = lambda x: x[1])
    distances
    A=[]
    B=[]

    for i in distances [1:6]:
A.append({new.iloc[i[0]].title})
    for i in range(len(distances [1:6])):
B.append(distances[i])
    for i in range(len(A)):
print("Predicted Movie:",A[i], B[i])        


# In[ ]:


search=input().lower()
recommend(search)


# In[ ]:


new_df.iloc[507].title


# In[ ]:


import pickle


# In[ ]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




