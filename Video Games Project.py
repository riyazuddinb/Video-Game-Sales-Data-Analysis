#!/usr/bin/env python
# coding: utf-8

# # Video Game Sales What Can We See From The Numbers ?
# 
# Video game is always related to our childhood. We played game when we're small and even when we're already an adult. But is the industry doing well these day ? We can analyze the video game sale dataset with graphs visualization to get some insight about that.
# 
# The dataset is taken from https://www.kaggle.com/rishidamarla/video-game-sales
# 
# Libraries used in project : 
# * [Pandas](https://pandas.pydata.org/) : a software library written for the Python programming language for data manipulation and analysis
# * [Numpy](https://numpy.org/) : a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
# * [Matplotlib](https://matplotlib.org/) : a plotting library for the Python programming language and its numerical mathematics extension NumPy.
# * [Seaborn](https://seaborn.pydata.org/) : a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
# 
# Thanks [Jovian](https://jovian.ml/) for the course project.
# 

# ### How to run the code
# 
# This is an executable [*Jupyter notebook*](https://jupyter.org) hosted on [Jovian.ml](https://www.jovian.ml), a platform for sharing data science projects. You can run and experiment with the code in a couple of ways: *using free online resources* (recommended) or *on your own computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing this notebook is to click the "Run" button at the top of this page, and select "Run on Binder". This will run the notebook on [mybinder.org](https://mybinder.org), a free online service for running Jupyter notebooks. You can also select "Run on Colab" or "Run on Kaggle".
# 
# 
# #### Option 2: Running on your computer locally
# 
# 1. Install Conda by [following these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Add Conda binaries to your system `PATH`, so you can use the `conda` command on your terminal.
# 
# 2. Create a Conda environment and install the required libraries by running these commands on the terminal:
# 
# ```
# conda create -n zerotopandas -y python=3.8 
# conda activate zerotopandas
# pip install jovian jupyter numpy pandas matplotlib seaborn opendatasets --upgrade
# ```
# 
# 3. Press the "Clone" button above to copy the command for downloading the notebook, and run it on the terminal. This will create a new directory and download the notebook. The command will look something like this:
# 
# ```
# jovian clone notebook-owner/notebook-id
# ```
# 
# 
# 
# 4. Enter the newly created directory using `cd directory-name` and start the Jupyter notebook.
# 
# ```
# jupyter notebook
# ```
# 
# You can now access Jupyter's web interface by clicking the link that shows up on the terminal or by visiting http://localhost:8888 on your browser. Click on the notebook file (it has a `.ipynb` extension) to open it.
# 

# ## Downloading the Dataset
# 
# Firstly We need to download the dataset to use. The link is already provided in the description above. You can also find a lot of interesting datasets on [Kaggle](https://www.kaggle.com/)

# In[1]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[2]:


dataset_url = 'https://www.kaggle.com/rishidamarla/video-game-sales' 


# The downloader will need to use ur username and apikey (generated in ur profile account on Kaggle) so firstly you should probably regis an account on Kaggle.

# In[3]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[4]:


data_dir = './video-game-sales'


# In[5]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[6]:


project_name = "data-analysis-of-video-game-sales"


# In[7]:


get_ipython().system('pip install jovian --upgrade -q')


# In[8]:


import jovian


# In[9]:


jovian.commit(project=project_name)


# ## Data Preparation and Cleaning
# 
# Firstly we should load the dataset into Pandas data frame and take a look what can we get with this dataset.
# 
# 

# In[10]:


import pandas as pd


# In[11]:


game_sales_df = pd.read_csv('./video-game-sales/Video_Games.csv')


# In[12]:


game_sales_df

Pretty cool we have 16719 rows equal to 16719 game titles here. We should probably check out the columns and info to see if this dataset is already workable
# In[13]:


game_sales_df.columns


# In[14]:


game_sales_df.info()


# Look at the info we can see that :
# - Not every game is rating and get critic score. 
# - Year of release | Platform doesnt match the name index. 
# 
# We should try removing nun object for a better dataframe.

# In[21]:


game_sales_df.drop(game_sales_df[game_sales_df.Year_of_Release.isnull()].index, inplace = True) #remove null value in Year of release column
game_sales_df.drop(game_sales_df[game_sales_df.Name.isnull()].index, inplace = True) #remove null value in Name column
game_sales_df.drop(game_sales_df[game_sales_df.Publisher.isnull()].index, inplace = True) #remove null value in Publisher column
game_sales_df.info()


# Ok that dataframe seems good enough. We should take a closer look at the description.

# In[18]:


game_sales_df.describe()


# - We have around 16416 game titles that was sold between 1980 and 2020. 
# - NA seems like the biggest market to sell game.
# - Sales are in millions 

# In[22]:


import jovian


# In[23]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# At first look the dataframe is already sorted by Global_Sales. But for a better viewer we should try creating a few graphs.
# 

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[25]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 13
matplotlib.rcParams['figure.figsize'] = (36, 20)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ### Total Sales Every Year

# First, We should see the total sales of games each year. It helps us know when video games are declining and when they are popular.

# In[26]:


sns.countplot('Year_of_Release', data = game_sales_df)
plt.title('Total Game Sales Each Year')
plt.show()


# Seems like we don't have much data from 2017 to 2020 let remove them and try using another graph for better view.

# In[28]:


# remove games that were released after 2016 
game_sales_df.drop(game_sales_df[game_sales_df.Year_of_Release > 2016].index, inplace = True)

sales_df = game_sales_df.groupby('Year_of_Release', as_index = False).sum()

x_axis = sales_df['Year_of_Release']
y_axis = sales_df['Global_Sales']

plt.figure(figsize=(20,10), dpi= 60)
plt.plot(x_axis, y_axis, label = 'Sales', color = 'green')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Total Game Sale Each Year')
plt.legend()
plt.show()


# ### Total Sale Comparison Between Region Area

# Let add other sales area as well like NA | EU | JP 

# In[30]:


na = sales_df['NA_Sales']
eu = sales_df['EU_Sales']
jp = sales_df['JP_Sales']
total = sales_df['Global_Sales']

plt.title('Sales Comparison Between Region And Global')
plt.plot(x_axis, total, label = 'Global')
plt.plot(x_axis, na, label = 'US')
plt.plot(x_axis, eu, label = 'EU')
plt.plot(x_axis, jp, label = 'JP')
plt.legend(bbox_to_anchor =(1, 1))


# We can see that the US is the largest market followed by the EU and JP. JP is pretty consistent and doesn't seem to be declined that much. In 2008 and 2009 video games were explored in popularity so we should take a look at the game list in these years.

# ### Top 10 Games and Platform in 2008 and 2009

# In[31]:


top_games_2008 = game_sales_df.loc[game_sales_df['Year_of_Release'] == 2008]
top_games_2008.sort_values('Global_Sales',ascending = False).head(10)


# In[32]:


top_games_2009 = game_sales_df.loc[game_sales_df['Year_of_Release'] == 2009]
top_games_2009.sort_values('Global_Sales',ascending = False).head(10)


# In 2008 and 2009, the most popular game was from Wii platform. That's pretty interesting let see the pie graph for platform (We should combine two dataframe as well)

# In[33]:


combine_list = top_games_2008.append(top_games_2009)
platform_counts = combine_list.Platform.value_counts()
platform_counts


# In[36]:


plt.figure(figsize=(24,12))
plt.title("Top 10 Platform in 2008 and 2009")
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# ### Top 10 Platform Overall

# In[37]:


top10_platforms = game_sales_df.Platform.value_counts().head(10)

plt.figure(figsize=(24,12))
plt.title("Top 10 platform of all time")
plt.pie(top10_platforms, labels=top10_platforms.index, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# PS2 still dominated for many years, truly the best selling console of all time.

# ### Top 10 Publishers

# In[38]:


top_publishers = game_sales_df.Publisher.value_counts().head(10)
top_publishers


# In[39]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
sns.barplot(top_publishers.index, top_publishers);


# ### Top 10 Genre

# In[40]:


top_genres = game_sales_df.Genre.value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(top_genres.index, top_genres);


# We should use Pie chart for this kind of thing. Since It can give you the percent of each genre as well.

# In[41]:


plt.figure(figsize=(24,12))
plt.title("Top 10 Genre")
plt.pie(top_genres, labels=top_genres.index, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# Let us save and upload our work to Jovian before continuing

# In[42]:


import jovian


# In[43]:


jovian.commit()


# ## Asking and Answering Questions
# 
# 

# #### Q1: How many games was sold in the US from 2000 to 2016 ? How does it compare to Global sale ? 

# In[96]:


game_sales_2000_to_2016 = game_sales_df[(game_sales_df['Year_of_Release'] >= 2000) & (game_sales_df['Year_of_Release'] <= 2016)]
total_sales_us = game_sales_2000_to_2016.NA_Sales.sum()
total_sales_jp = game_sales_2000_to_2016.JP_Sales.sum()
total_sales_eu = game_sales_2000_to_2016.EU_Sales.sum()
total_sales_others = game_sales_2000_to_2016.Other_Sales.sum()

data = [['US', total_sales_us],['JP', total_sales_jp],['Others', total_sales_others],['EU', total_sales_eu]]
df = pd.DataFrame(data, columns = ['Name', 'Sales']) 


# In[97]:


plt.figure(figsize=(24,12))
plt.title("US Market Share")
plt.pie(df.Sales, labels=df.Name, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# #### Q2: Assume We want to join the game industry and target the US market. Which genre should we try to make ?
# 
# After taking a look at the top 10 genre chart we can see that Action is the most popular genre.But we should check out the top genre in the US first then compare it to other regions.

# In[53]:


# sort_values sort the data frame with the correct column name you can specific ascending true | false for 
# head (number) return the number of row 
# we get 1000 result and try get percent of genre that's popular in the US 

top_1000_us = game_sales_df.sort_values('NA_Sales',ascending = False).head(1000)
top_1000_us


# In[55]:


# value_counts : return a Series containing counts of unique values
top_1000_us_genre = top_1000_us.Genre.value_counts()

plt.figure(figsize=(24,12))
plt.title("Top 10 Genre US")
plt.pie(top_1000_us_genre, labels=top_1000_us_genre.index, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# Looking at the chart we can safely assume that Action and Shooter are really popular in the US. So for a better chance of success if we want to make games we should create a game combined between Action and Shooter like Overwatch!

# #### Q3: Who is the top publisher in Japan ? What game is their best seller and did they focus in some specific genre or just publish whatever they think will be popular ?

# Firstly, We should find out who is the current top publisher in Japan. Then we can calculate the genre percent of their published games and create a chart. Looking at the chart can give us a better view for the answer.

# In[57]:


top_publishers = game_sales_df.groupby('Publisher').sum()
top_publishers_jp = top_publishers.sort_values('JP_Sales',ascending = False).head(10)
top_publishers_jp


# So the top publisher in Japan is Nintendo with 457 millions sales. Next let see what is their best seller.

# In[58]:


top_games_nintendo = game_sales_df.loc[game_sales_df['Publisher'] == 'Nintendo'].sort_values('JP_Sales',ascending = False).head(10)
top_games_nintendo


# The best seller game of Nintendo in Japan is Pokemon Red/Pokemon Blue which sold 10.22 millions copy.

# In[64]:


top_genre_nintendo = top_games_nintendo.Genre.value_counts()

plt.figure(figsize=(24,12))
plt.title("Top 10 Genre Nintendo")
plt.pie(top_genre_nintendo, labels=top_genre_nintendo.index, autopct='%1.1f%%', startangle=180);
plt.legend(loc = 2,fontsize  = 10, bbox_to_anchor = (1, 1), ncol = 2)


# Their focus seems like Role-Playing (Pokemon series) and Platform (Mario). 

# #### Q4: Make a chart to display how the top trending game (genre) in 2008 was doing up to 2015 and how the top trending in 2015 was doing before (to 2008)

# In[66]:


top_game_2008 = game_sales_df.loc[game_sales_df['Year_of_Release'] == 2008].sort_values('Global_Sales',ascending = False).head(1)
top_game_2008


# In[70]:


top_game_2015 = game_sales_df.loc[game_sales_df['Year_of_Release'] == 2015].sort_values('Global_Sales',ascending = False).head(1)
top_game_2015


# so We now have 2 different genre : Racing and Shooting. Let get all the games released between 2008 and 2015.

# In[78]:


games_list =  game_sales_df[(game_sales_df['Year_of_Release'] >= 2008) & (game_sales_df['Year_of_Release'] <= 2015)]
games_list = games_list.groupby(['Genre', 'Year_of_Release'], as_index = False).sum()
games_list


# In[80]:


racing_games_list = games_list.loc[games_list['Genre'] == 'Racing']
x = racing_games_list['Year_of_Release']
y = racing_games_list['Global_Sales']

plt.figure(figsize=(20,10), dpi= 60)
plt.plot(x, y, label = 'Sales', color = 'green')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Racing Game Trending')
plt.legend()
plt.show()


# In[82]:


shooting_games_list = games_list.loc[games_list['Genre'] == 'Shooter']
x = shooting_games_list['Year_of_Release']
y = shooting_games_list['Global_Sales']

plt.figure(figsize=(20,10), dpi= 60)
plt.plot(x, y, label = 'Sales', color = 'green')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Shooter Game Trending')
plt.legend()
plt.show()


# Let us save and upload our work to Jovian before continuing.

# In[98]:


import jovian


# In[99]:


jovian.commit()


# ## Inferences and Conclusion
# 
# This dataset helps us understand a lot of things about the game market. Here is just a small possible analytics from the data, we haven't even used the other columns like Critic, User, Developer or Rating yet. After just some questions above I realized the importance of data on how we can change it from number to possibility. The number doesn't lie they provide use true information that can be used to improve. 

# In[30]:


import jovian


# In[31]:


jovian.commit()


# ## References and Future Work
# 
# For my future work I planned to :
# - Implement machine training to predict the next trending genre.
# - Combine this dataset with specific data about games rating on other websites for evaluation of publishers or companies.
# - Making my own dataset by crawing data from website.
# - Learn more about data sciences in general to improve my understanding.

# In[100]:


import jovian


# In[101]:


jovian.commit()


# In[ ]:




