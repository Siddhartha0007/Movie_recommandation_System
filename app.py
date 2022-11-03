# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:01:02 2022

@author: Siddhartha-PC
"""

###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import scipy
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
## Hyperopt modules
#from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
#from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from yellowbrick.classifier import PrecisionRecallCurve
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import re
import sys
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv 
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#lottie animations
import time
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

#nltk libreries

import io
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
#from surprise import SVD,Reader,Dataset
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

###############################################Data Processing###########################
# Importing Data and Pickle file
df = pd.read_csv(r'./movies.csv')
ratings = pd.read_csv(r'./ratings2.csv')
links = pd.read_csv(r'./links_small.csv')
links.dropna(subset='tmdbId',inplace=True)
ratings = pd.read_csv(r'./ratings_small.csv')

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
project_url_1="https://assets9.lottiefiles.com/packages/lf20_bzgbs6lx.json"
project_url_2="https://assets6.lottiefiles.com/packages/lf20_eeuhulsy.json"
report_url="https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json"
about_url="https://assets2.lottiefiles.com/packages/lf20_k86wxpgr.json"

about_1=load_lottieurl(about_url)
report_1=load_lottieurl(report_url)
project_1=load_lottieurl(project_url_1)
project_2=load_lottieurl(project_url_2)

lottie_download = load_lottieurl(lottie_url_download)

#st_lottie(lottie_hello, key="hello")


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)
















###############################################Streamlit Main###############################################

def main():
    # set page title
    
    
            
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title= None,options=["Home", "Project","Report" ,"About"], icons=["house-door", "cast","clipboard","file-person"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "cyan"},"icon": {"color": "#6c3483", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#ff5733", },"nav-link-selected":{"background-color":"#2874a6"},},)
    
    #horizontal Home selected
    if selected == "Home":
        #image= Image.open("home_img.jpg")
        #st.image(image,use_column_width=True)
        #st.title("Home") 
        col1, col2 = st.columns(2)
        with col1:
            lottie_url_hello1 = "https://assets2.lottiefiles.com/packages/lf20_zvcyhdqv.json"
            lottie_hello1 = load_lottieurl(lottie_url_hello1)
            st_lottie(lottie_hello1, key="hello",)
        with col2:
            image= Image.open("home1.jpeg")
            st.image(image,use_column_width=True)
        st.sidebar.title("Home")        
        with st.sidebar:
            lottie_url_hello2 = "https://assets10.lottiefiles.com/packages/lf20_zdm1abxk.json"
            lottie_hello2 = load_lottieurl(lottie_url_hello2)
            st_lottie(lottie_hello2, key="hello2",)
            #image= Image.open("Home1.png")
            #add_image=st.image(image,use_column_width=True)
            
            
        
        def header(url):
            st.sidebar.markdown(f'<p style="background-color:#a569bd ;color:white;font-size:15px;border-radius:1%;">{url}', unsafe_allow_html=True)    
        html_45=""" A Quick Youtube Video for understanding the MOVIE RECOMMENDATION SYSTEMS for Educational Purpose ."""
        
        #st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        with st.sidebar:
            #image= Image.open("Home1.png")
            st.write('Author@ Siddhartha Sarkar')
            st.write('Data Scientist ')
        st.balloons()
        header(html_45)
        st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> 🎥 MOVIE RECOMMENDATION SYSTEMS  🎥 Using Machine Learning</h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color:royalblue ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
		 Why Recommender Systems ? 
         
             The Era of Recommender Systems :

    The rapid growth of data collection has led to a new era of information. 
    Data is being used to create more efficient systems and this is where Recommendation Systems come into play. 
    Recommendation Systems are a type of information filtering systems as they improve the quality of search results 
    and provides items that are more relevant to the search item or are realted to the search history of the user.

    They are used to predict the rating or preference that a user would give to an item. 
    Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest 
    products to customers, YouTube uses it to decide which video to play next on autoplay, and 
    Facebook uses it to recommend pages to like and people to follow. Moreover, companies like Netflix and 
    Spotify depend highly on the effectiveness of their recommendation engines for their business and sucees.


        
		  """
        
		
        header(html_temp11)
        def header(url):
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp12 = """
		 Recommendation System:<br>
    A recommendation system is an artificial intelligence or AI algorithm, usually associated with machine learning, 
    that uses Big Data to suggest or recommend additional products to consumers. These can be based on various criteria, 
    including past purchases, search history, demographic information, and other factors. Recommender systems are highly 
    useful as they help users discover products and services they might otherwise have not found on their own.
    Recommender systems are trained to understand the preferences, previous decisions, and characteristics of people and 
    products using data gathered about their interactions. These include impressions, clicks, likes, and purchases. 
    Because of their capability to predict consumer interests and desires on a highly personalized level, recommender 
    systems are a favorite with content and product providers. They can drive consumers to just about any product or 
    service that interests them, from books to videos to health classes to clothing.   
		  """
        
		
        header(html_temp12)
        
        col3,col4=st.columns(2)
        with col3:
            image= Image.open("differ_rec_eng.jpg")
            st.image(image,use_column_width=False)
        with col4:
            image= Image.open("index.png")
            st.image(image,use_column_width=False)
        
        def header(url):
            
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp13 = """
		 There are basically three types of recommender systems:-<br>
    🎥Demographic Filtering(Popularity Based)- They offer generalized recommendations to every user, based on movie popularity and/or genre. 
    The System recommends the same movies to users with similar demographic features. Since each user is different , 
    this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular 
    and critically acclaimed will have a higher probability of being liked by the average audience.<br>
    🎥Content Based Filtering- They suggest similar items based on a particular item. This system uses item metadata, 
    such as genre, director, description, actors, etc. for movies, to make these recommendations. 
    The general idea behind these recommender systems is that if a person liked a particular item, he or she will also 
    like an item that is similar to it.<br>
    🎥Collaborative Filtering- This system matches persons with similar interests and 
    provides recommendations based on this matching. Collaborative filters do not require item metadata like its 
    content-based counterparts.<br>
    🎥Hybrid Filtering-<br><br>Defination of Different Filtering: How It Works<br><br>  
     🎥Demographic Filtering -
     we need a metric to score or rate movie Calculate the score for every movie Sort the scores and recommend the best 
     rated movie to the users. We can use the average ratings of the movie as the score but using this won't be fair 
     enough since a movie with 8.9 average rating and only 3 votes cannot be considered better than the movie with 7.8 
     as as average rating but 40 votes. So, I'll be using IMDB's weighted rating (wr) formula:-
     v is the number of votes for the movie; m is the minimum votes required to be listed in the chart; R is the average 
     rating of the movie; And C is the mean vote across the whole report We already have v(vote_count) and R 
     (vote_average) and C can be calculated as <br>
     🎥Content Based Filtering
     In this recommender system the content of the movie (overview, cast, crew, keyword, tagline etc) is used to find 
     its similarity with other movies. Then the movies that are most likely to be similar are recommended.
     With this matrix in hand, we can now compute a similarity score. There are several candidates for this; 
     such as the euclidean, the Pearson and the cosine similarity scores. There is no right answer to which score is the 
     best. Different scores work well in different scenarios and it is often a good idea to experiment with different 
     metrics. image.png We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity 
     between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy 
     and fast to calculate. <br>
     🎥Collaborative Filtering
     Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are 
     close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
     Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a 
     user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that 
     movie, regardless of who she/he is.
     Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to 
     Movie Watchers. It is basically of two types:-
     User based filtering- These systems recommend products to a user that similar users have liked. For measuring the 
     similarity between two users we can either use pearson correlation or cosine similarity. This filtering technique 
     can be illustrated with an example. In the following matrixes, each row represents a user, while the columns 
     correspond to different movies except the last one which records the similarity between that user and the target 
     user. Each cell represents the rating that the user gives to that movie. Assume user E is the target. Since user 
     A and F do not share any movie ratings in common with user E, their similarities with user E are not defined in 
     Pearson Correlation. Therefore, we only need to consider user B, C, and D. Based on Pearson Correlation, we can 
     compute the following similarity.
     From the above table we can see that user D is very different from user E as the Pearson Correlation between 
     them is negative. He rated Me Before You higher than his rating average, while user E did the opposite. Now, 
     we can start to fill in the blank for the movies that user E has not rated based on other users.
     Although computing user-based CF is very simple, it suffers from several problems. One main issue is that users’ 
     preference can change over time. It indicates that precomputing the matrix based on their neighboring users may 
     lead to bad performance. To tackle this problem, we can apply item-based CF.
     Item Based Collaborative Filtering - Instead of measuring the similarity between users, the item-based CF 
     recommends items based on their similarity with the items that the target user rated. Likewise, the similarity 
     can be computed with Pearson Correlation or Cosine Similarity. The major difference is that, with item-based 
     collaborative filtering, we fill in the blank vertically, as oppose to the horizontal manner that user-based CF does.
     The following table shows how to do so for the movie Me Before You. It successfully avoids the problem posed by 
     dynamic user preference as item-based CF is more static. However, several problems remain for this method. 
     First, the main issue is scalability. The computation grows with both the customer and the product. 

		  """      
		
        header(html_temp13)
        
        
# =============================================================================
#         def plot11():
#             
#             import plotly.graph_objects as go
# 
#             values = [['movie_id', 'cast', 'crew', 'bruises', 'odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat','class'], #1st col
#             ["bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s",
#             "fibrous=f,grooves=g,scaly=y,smooth=s",
#             "brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y",
#             "bruises=t,no=f ",
#             "  almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s",
#             " attached=a,descending=d,free=f,notched=n",
#             "  close=c,crowded=w,distant=d ",
#             "broad=b,narrow=n",
#             " black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y",
#             " enlarging=e,tapering=t ",
#             "  bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? ",
#             "fibrous=f,scaly=y,silky=k,smooth=s",'fibrous=f,scaly=y,silky=k,smooth=s','brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y',
#             " brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y",'partial=p,universal=u','brown=n,orange=o,white=w,yellow=y',
#             'none=n,one=o,two=t','cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z ',
#             'black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y ','abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y',
#             ' grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d','Different Class of Mushrooms']]
# 
#             fig = go.Figure(data=[go.Table(
#             columnorder = [1,2],
#             columnwidth = [80,400],
#             header = dict(
#             values = [['<b>Columns of<br>  Dataset </b>'],
#                   ['<b>Attribute Information</b>']],
#             line_color='red',
#             fill_color='royalblue',
#             align=['left','center'],
#             font=dict(color='black', size=12),
#             height=40
#                   ),
#             cells=dict(
#             values=values,
#             line_color='red',
#             fill=dict(color=['pink', 'lightskyblue']),
#             font=dict(color='black', size=12),
#             align=['left', 'center'],
#               font_size=12,
#             height=20)
#               )
#            ])
#             return fig
# 
#         p11=plot11()
# =============================================================================
        def header(url):
            st.markdown(f'<p style="background-color:#d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        
        html_temp1133 = """       
                 About The DataSet : <br>
                 The first dataset contains the following features:-<br>
                  movie_id - A unique identifier for each movie.<br>
                  cast - The name of lead and supporting actors.<br>
                  crew - The name of Director, Editor, Composer, Writer etc.<br>
                  The second dataset has the following features:-<br>
                  budget - The budget in which the movie was made.<br>
                  genre - The genre of the movie, Action, Comedy ,Thriller etc.<br>
                  homepage - A link to the homepage of the movie.<br>
                  id - This is infact the movie_id as in the first dataset.<br>
                  keywords - The keywords or tags related to the movie.<br>
                  original_language - The language in which the movie was made.<br>
                  original_title - The title of the movie before translation or adaptation.<br>
                  overview - A brief description of the movie.<br>
                  popularity - A numeric quantity specifying the movie popularity.<br>
                  production_companies - The production house of the movie.<br>
                  production_countries - The country in which it was produced.<br>
                  release_date - The date on which it was released.<br>
                  revenue - The worldwide revenue generated by the movie.<br>
                  runtime - The running time of the movie in minutes.<br>
                  status - "Released" or "Rumored".<br>
                  tagline - Movie's tagline.<br>
                  title - Title of the movie.<br>
                  vote_average - average ratings the movie recieved.<br>
                  vote_count - the count of votes recieved."""
        #st.plotly_chart(p11)
        header(html_temp1133)
# =============================================================================
#         def plot12():
#             import plotly.figure_factory as ff
#             df_sample = mushroom_data.iloc[0:10,0:9]
#             colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
#             font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
#             fig =  ff.create_table(df_sample,colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])
#             fig.show()
#             return fig
#         p12=plot12()
#         st.write("Data Table")
#         st.plotly_chart(p12)
# =============================================================================

        def header(url):
            st.markdown(f'<p style="background-color:#d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp111 = """
		About The Project :<br>
        In this project,I tried to build Recommendation Systems ,Filtering systems to improve the quality of 
        search results and provides items that are more relevant to the search item or are realted to the 
        search history of the user.


        """
        header(html_temp111)
# =============================================================================
#         st.markdown("""
#                 #### Tasks Perform by the app:
#                 + App covers the most basic Machine Learning task of  Analysis, Correlation between variables,project report.
#                 + Machine Learning on different Machine Learning Algorithms, building different models and lastly  prediction.
#                 
#                 """)
# =============================================================================
                
    #Horizontal About selected
    if selected == "About":
        #st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About-Us-PNG-Isolated-Photo.png")
            add_image=st.image(image,use_column_width=True)
        
        st_lottie(about_1,key='ab1')
        #image2= Image.open("about.jpg")
        #st.image(image2,use_column_width=True)
        #st.sidebar.write("This Youtube Video Shows and Describes Different Kind Of Mushrooms for Learning Purpose ")
        #st.sidebar.video('https://www.youtube.com/watch?v=6PNq6paMBXU')
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">This is a 🎥 MOVIE RECOMMENDATION SYSTEMS  🎥 Project</h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:30px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_99   =  """
        In this project,I tried to build Recommendation Systems,Filtering systems to improve the quality of 
        search results and provides items that are more relevant to the search item or are realted to the 
        search history of the user    """
        header(html_99)
        
        st.sidebar.markdown("""
                    #### + Project Done By :        
                    #### @Author Mr. Siddhartha Sarkar
                    
        
                    """)
        st.snow()
        
        #st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    #Horizontal Project_Report selected
    if selected == "Report":
        #report_1
        st.title("Profile Report")
        st.sidebar.title("Project_Profile_Report")
        
        with st.sidebar:
            st_lottie(report_1, key="report1")
            #image= Image.open("report_project.png")
            #add_image=st.image(image,use_column_width=True)
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Simple EDA App with Streamlit Components</h1>
		</div>  """
        
		
        components.html(html_temp)
        html_temp1 = """
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1500px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
			  <div class="text">Caption Text</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text">Caption Two</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text">Caption Three</div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			"""
        components.html(html_temp1)
        st.sidebar.title("Navigation")
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file= st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.table(df.head(10))
                m = st.markdown("""
                       <style>
                    div.stButton > button:first-child {
                  background-color: #0099ff;
                      color:#ffffff;
                                   }
                    div.stButton > button:hover {
                     background-color: #00ff00;
                         color:#ff0000;
                          }
                    </style>""", unsafe_allow_html=True)

                if st.button("Generate Profile Report"):
                    profile= ProfileReport(df)
                    st_profile_report(profile)
            
            
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file = st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df =  pd.read_csv(data_file)
                st.dataframe(df.head(10))
                m = st.markdown("""
                            <style>
                         div.stButton > button:first-child {
                                 background-color: #0099ff;
                                      color:#ffffff;
                                        }
                            div.stButton > button:hover {
                               background-color: #00ff00;
                                       color:#ff0000;
                                             }
                        </style>""", unsafe_allow_html=True)

                if st.button("Generate Sweetviz Report"):
                    # Normal Workflow
                    report = sv.analyze(df)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html") 
               			
		      
    #Horizontal Project selected
    if selected == "Project":
            df = pd.read_csv(r'./movies.csv')
            ratings = pd.read_csv(r'./ratings2.csv')
            links = pd.read_csv(r'./links_small.csv')
            links.dropna(subset='tmdbId',inplace=True)
            ratings = pd.read_csv(r'./ratings_small.csv')
            
            with st.sidebar:
                st_lottie(project_1, key="pro1")                
            import time                
            st_lottie(project_2, key="pro22")
            st.title("Projects")              
            #image2= Image.open("project11.jpeg")
            #st.image(image2,use_column_width=True)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            with st.sidebar:
                
                menu_Pre_Exp = option_menu("App Gallery", ['Dataset Info',"Exploratory Data Analysis", "Get Recommendations"],
                         icons=[ 'pc-display-horizontal', 'kanban','collection-play-fill'],
                         menu_icon="app-indicator", default_index=0,orientation="vertical",
                         styles={
        "container": {"padding": "5!important", "background-color": "#c39bd3"},
        "icon": {"color": "#2980b9", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ec7063"},
        "nav-link-selected": {"background-color": "green"}, } )
            
            
            #menu_list1 = ['Exploratory Data Analysis',"Prediction With Machine Learning"]
            #menu_Pre_Exp = st.sidebar.radio("Menu For Prediction & Exploratoriy", menu_list1)
            
            #EDA On Document File
            if  menu_Pre_Exp == 'Exploratory Data Analysis' : # and selected == "Projects"
                    st.title('Exploratory Data Analysis')
                    df = pd.read_csv(r'./movies.csv')
                    st.write("Table")
                    st.write(df.head(10))
                    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                    ratings = pd.read_csv(r'./ratings2.csv')
                    m = st.markdown("""
                                  <style>
                                  div.stButton > button:first-child {
                                   background-color: #0099ff;
                                       color:#ffffff;
                                                         }
                                   div.stButton > button:hover {
                                   background-color: #00ff00;
                                   color:#ff0000;
                                                   }
                                  </style>""", unsafe_allow_html=True)
                    submit = st.button(label='Generate Visualizations')
                    if submit:
                        fig = plt.figure(figsize=(8, 5))
                        sns.distplot(df['budget'])
                        plt.title('Budget', weight='bold')
                        st.pyplot(fig)
                        st.write("The distribution of movie budgets shows an exponential decay.")


                        st.title('Popularity')
                        fig = plt.figure(figsize=(8, 5))
                        df['popularity'].plot(logy=True, kind='hist')
                        plt.xlabel('popularity')
                        st.pyplot(fig)
                        st.write(
                            "As the popularity score it seems to be extremely right skewed data with the mean of 2.7 and maximum reaching upto 294 and the 75% percentile is at 3.493 and almost all the data below 75%")

                        st.title('Overview-Wordcloud')
                        image = Image.open(r'overview.png')
                        st.image(image)
                        st.write(
                            "Life is the most commonly used word in Overview,followed by 'one' and 'find' are the most Movie Blurgs.Together with Love, Man and Girl, these wordclouds give us a pretty good idea of the most popular themes present in movies.")

                        st.title('Title-Wordcloud')
                        image = Image.open(r'title.png')
                        st.image(image)
                        st.write(
                            "As we can see 'LOVE' the title is common in most of the Movie title followed by 'LIFE','GIRL','MAN' and 'NIGHT'")
                        st.title('Genres-Wordcloud')
                        image = Image.open(r'genres.png')
                        st.image(image)
                        
                        
                        plt.title('Released_year vs movies', weight='bold')
                        year_df = pd.DataFrame(df['release_year'].value_counts())
                        year_df['year'] = year_df.index
                        year_df.columns = ['number', 'year']
                        fig = plt.figure(figsize=(12, 5))
                        sns.barplot(x='year', y='number', data=year_df.iloc[1:20])
                        st.pyplot(fig)
                        st.write("By the Relaesed_year we inferetiate that Most number of movies released in 2006")

                        fig = plt.figure(figsize=(8, 5))
                        ax = sns.distplot(df['vote_average'])
                        plt.title('Vote Average', weight='bold')
                        plt.xlabel('Vote_Average', weight='bold')
                        plt.ylabel('Density', weight='bold')
                        st.pyplot(fig)
                        st.write(
                            "There is a very small correlation between Vote Count and Vote Average. A large number of votes on a particular movie does not necessarily imply that the movie is good.")

                        fig = plt.figure(figsize=(8, 5))
                        ax = sns.distplot(df[(df['runtime'] < 300) & (df['runtime'] > 0)]['runtime'])
                        plt.title('Runtime', weight='bold')
                        plt.xlabel('Runtime', weight='bold')
                        plt.ylabel('Density', weight='bold')
                        st.pyplot(fig)
                        st.write("Here we count that runtime is less than 300 but greater than 0 ")

                        fig = plt.figure(figsize=(10, 10))
                        df['genres'].value_counts()[:20].plot(kind='barh')
                        plt.title("Movies genres ", fontsize=20)
                        plt.ylabel("movie genres", fontsize=20)
                        plt.xlabel("count", fontsize=20)
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=17)
                        st.pyplot(fig)
                        st.write(
                            "As per the above count plot it seems there is highest no.of TV series genres are Drama as compared to the other TV series. ")

                        fig = plt.figure(figsize=(15, 7))
                        df['cast'].value_counts()[:20].plot(kind='barh')
                        plt.title("Movies cast ", fontsize=20)
                        plt.ylabel("movie cast", fontsize=20)
                        plt.xlabel("Count", fontsize=20)
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=17)
                        plt.show()
                        st.pyplot(fig)
                        st.write("As per the above count plot of cast it seems GeorgesMeeleis has acted in many TV series")

                        fig = plt.figure(figsize=(15, 7))
                        df['crew'].value_counts()[:20].plot(kind='barh')
                        plt.title("Movies crew ", fontsize=20)
                        plt.ylabel("movies crew", fontsize=20)
                        plt.xlabel("Count", fontsize=20)
                        plt.xticks(fontsize=14)
                        plt.yticks(fontsize=17)
                        plt.show()
                        st.pyplot(fig)
                        st.write("As per the above count plot of crew it seems JohnFord has directed many TV series")

                        fig = plt.figure(figsize=(20, 20))

                        st.title("Histogram ")

                        fig = plt.figure(figsize=(20, 20))

                        for i, col in enumerate(numerical_cols[:-1]):
                            plt.subplot(10, 3, i + 1)
                            plt.hist(df[col])
                            plt.xlabel(col)
                            plt.subplots_adjust(left=0.1,
                                                bottom=0.1,
                                                right=0.9,
                                                top=0.9,
                                                wspace=0.4,
                                                hspace=0.4)
                        plt.show()
                        st.pyplot(fig)

                        st.title("distribution")
                        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
                        fig = plt.figure(figsize=(20, 20))

                        for i, col in enumerate(numerical_cols[:-1]):
                            plt.subplot(10, 3, i + 1)
                            sns.distplot(df[col], bins=20, kde=True, rug=True)
                            plt.xlabel(col)
                            plt.subplots_adjust(left=0.1,
                                                bottom=0.1,
                                                right=0.9,
                                                top=0.9,
                                                wspace=0.4,
                                                hspace=0.4)
                        st.pyplot(fig)

                        st.title("Box plot")
                        fig = plt.figure(figsize=(20, 20))

                        for i, col in enumerate(numerical_cols[:-1]):
                            plt.subplot(10, 3, i + 1)
                            sns.boxplot(df[col])
                            plt.xlabel(col)
                            plt.subplots_adjust(left=0.1,
                                                bottom=0.1,
                                                right=0.9,
                                                top=0.9,
                                                wspace=0.4,
                                                hspace=0.4)
                        plt.show()
                        st.pyplot(fig)

                        st.write(
                            "1)Very few TV series has generated the higher revenue as shown in the histogram. 2)The Vote average of the TV series between range 3 to 9 as shown in the bar plot. 3)The Vote average column is normally distrubuted as shown in the distribution plot 4)The runtime column has right tail which means it is right skewed as per the distribution plot.")

                        fig = plt.figure(figsize=(10, 10))
                        year_runtime = df[df['release_year'] != 'NaT'].groupby('release_year')['runtime'].mean()
                        plt.plot(year_runtime.index, year_runtime)
                        plt.xticks(np.arange(1900, 2024, 10.0))
                        plt.title('Runtime vs Year_trend', weight='bold')
                        plt.xlabel('Year', weight='bold')
                        plt.ylabel('Runtime in min', weight='bold')
                        st.pyplot(fig)
                        st.write(
                            "As we can inference that trends go down on 1917 till 50 min and gain it increse upto 110 almost the ranges lies 90 to 110")

                        st.title("Production_countries vs revenue")
                        fig = plt.figure(figsize=(17, 5))
                        plt.subplot(1, 2, 1)
                        sns.barplot(data=df.head(20), x='revenue', y='production_countries')
                        st.pyplot(fig)
                        st.write(
                            "From the revenue vs production countries plot United Kingdom and United States of America occupy the 1st position")

                        fig = plt.figure(figsize=(10, 10))
                        axis1 = sns.barplot(x=df['vote_average'].head(10), y=df['title'].head(10), data=df)
                        plt.xlim(4, 10)
                        plt.title('Best Movies by average votes', weight='bold')
                        plt.xlabel('Weighted Average Score', weight='bold')
                        plt.ylabel('Movie Title', weight='bold')
                        st.pyplot(fig)
                        st.write("By the vote average we inferetiate that 'Toy Story'occupied the 1st position")

                        scored_df = ratings.sort_values('rating', ascending=False)
                        fig = plt.figure(figsize=(10, 10))
                        ax = sns.barplot(x=scored_df['rating'].head(10), y=scored_df['title'].head(10), data=scored_df, palette='deep')
                        plt.title('Best Rated & Most Popular Blend', weight='bold')
                        plt.xlabel('Score', weight='bold')
                        plt.ylabel('Movie Title', weight='bold')
                        st.pyplot(fig)
                        st.write("This are the top 10 movie title recieved 5 ratings")

                        fig = plt.figure(figsize=(10, 10))
                        ax = sns.barplot(x=df['popularity'].head(10), y=df['title'].head(10), data=df)
                        plt.title('Most Popular by votes', weight='bold')
                        plt.xlabel('Score of popularity', weight='bold')
                        plt.ylabel('Movie Title', weight='bold')
                        st.pyplot(fig)
                        st.write(
                            "From the popularity based ,we inferentiate that 'Toy story'  occupied the 1st position followed by 'Heat' and 'Jumaji' respectively.")

                        plt.title('Production_countries')
                        fig = plt.figure(figsize=(10, 10))
                        df['production_countries'].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
                        st.pyplot(fig)
                        st.write("As per the pie plot it seems USA has high production rate of making TV series")

                        fig = plt.figure(figsize=(10, 5))
                        s = ratings.sort_values(['rating'], ascending=False)[:20]
                        plt.title('top movies by average ratings')
                        sns.barplot(y='title', x='rating', data=s)
                        st.pyplot(fig)
                        st.write('Phantom of paradise is the top rated movie')

                    
                    
# =============================================================================
#                     menu_list2 = ['None', 'Basic_Statistics','Basic_Plots','Interactive_plots','Multi_level_SunBurst_plots','Feature Engineering']
#                     menu_Exp = st.sidebar.radio("Menu EDA", menu_list2)
# =============================================================================


            if  menu_Pre_Exp == "Get Recommendations" : #and selected == "Projects"
                st.title('Get Recommendations')
                
                def get_poster_url(id):
                    API_key = "a9940390778cc2fd7f3ee153bcec4d99"
                    URL = f"https://api.themoviedb.org/3/movie/{id}?api_key=a9940390778cc2fd7f3ee153bcec4d99"
                    PosterDB = requests.get(URL)
                    todos = json.loads(PosterDB.text)
                    path = todos['poster_path']
                    url_to_poster = 'https://image.tmdb.org/t/p/w500' + path
                    return url_to_poster

                
                select = st.selectbox(label='Select the type of recommendation',options=['Popularity based recommendations','Content based recommendations','Item based Collaborative filtering','User based Collaborative filtering'])
                if select == 'Popularity based recommendations':
                    v = df['vote_count']
                    R = df['vote_average']
                    C = df['vote_average'].mean()
                    m = df['vote_count'].quantile(0.70)
                    df['weighted_average'] = ((R * v) + (C * m)) / (v + m)
                    scaler = MinMaxScaler()
                    movies_scaled = scaler.fit_transform(df[['weighted_average','popularity']])
                    movies_tf = pd.DataFrame(movies_scaled,columns=['weighted_average','popularity'])
                    df[['weight_average_tf', 'popularity_tf']] = movies_tf
                    df['score'] = df['weight_average_tf']*0.5 + df['popularity_tf']*0.5
                    df_pop = df.sort_values(['score'],ascending=False)
                    m = st.markdown("""
                                       <style>
                               div.stButton > button:first-child {
                             background-color: #0099ff;
                                  color:#ffffff;
                                                 }
                                 div.stButton > button:hover {
                                    background-color: #00ff00;
                                          color:#ff0000;
                                                   }
                                  </style>""", unsafe_allow_html=True)

                    submit = st.button('Get Recommendations based on popularity')
                    if submit:

                        col1, col2, col3, col4, col5 = st.columns(5)
                        col6,col7,col8,col9,col10 = st.columns(5)
                        with col1:

                            st.image(get_poster_url(df_pop['id'].iloc[0]),caption=df_pop['title'].iloc[0],width=150)
                        with col2:

                            st.image(get_poster_url(df_pop['id'].iloc[1]),caption=df_pop['title'].iloc[1],width=150)

                        with col3:

                            st.image(get_poster_url(df_pop['id'].iloc[2]),caption=df_pop['title'].iloc[2],width=150)
                        with col4:

                            st.image(get_poster_url(df_pop['id'].iloc[3]),caption=df_pop['title'].iloc[3],width=150)
                        with col5:

                            st.image(get_poster_url(df_pop['id'].iloc[4]),caption=df_pop['title'].iloc[4],width=150)
                        with col6:

                            st.image(get_poster_url(df_pop['id'].iloc[5]),caption=df_pop['title'].iloc[5],width=150)
                        with col7:

                            st.image(get_poster_url(df_pop['id'].iloc[6]),caption=df_pop['title'].iloc[6],width=150)
                        with col8:

                            st.image(get_poster_url(df_pop['id'].iloc[7]),caption=df_pop['title'].iloc[7],width=150)
                        with col9:

                            st.image(get_poster_url(df_pop['id'].iloc[8]),caption=df_pop['title'].iloc[8],width=150)
                        with col10:

                            st.image(get_poster_url(df_pop['id'].iloc[9]),caption=df_pop['title'].iloc[9],width=150)
                    else:
                        print('Error')



                elif select == 'Content based recommendations':
                    movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
                    df_c = df[df['id'].isin(movie_id)]
                    df_c.dropna(inplace=True)
                    df_c.drop_duplicates(subset='title', inplace=True)
                    user = st.selectbox('Please select a movie to get recommendations', options=df_c['title'].tolist())
                    m = st.markdown("""
                                             <style>
                                  div.stButton > button:first-child {
                               background-color: #0099ff;
                                   color:#ffffff;
                                          }
                                     div.stButton > button:hover {
                                    background-color: #00ff00;
                                       color:#ff0000;
                                              }
                                   </style>""", unsafe_allow_html=True)

                    submit = st.button('Get recommendations based on content')

                    tf = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 6), stop_words='english', analyzer='word')
                    tf_idf = tf.fit_transform(df_c['overview'])
                    sigmoid = sigmoid_kernel(tf_idf, tf_idf)
                    indices = pd.Series(df_c['overview'].index, index=df_c['title'])


                    def get_rec(title, sigmoid=sigmoid):
                        idx = indices[title]
                        sig_scores = list(enumerate(sigmoid[idx]))
                        sig_scores1 = sorted(sig_scores, key=lambda x: x[1], reverse=True)
                        sig_scores2 = sig_scores1[1:11]
                        movie_indices = [i[0] for i in sig_scores2]
                        return df_c['title'].iloc[movie_indices]


                    def get_id(title):
                        idx = indices[title]
                        sig_scores = list(enumerate(sigmoid[idx]))
                        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
                        sig_scores = sig_scores[1:11]
                        movie_indices = [i[0] for i in sig_scores]
                        return df_c['id'].iloc[movie_indices]


                    if submit:
                        rec = get_rec(user)
                        ids = get_id(user)
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col6, col7, col8, col9, col10 = st.columns(5)
                        with col1:
                            st.image(get_poster_url(ids.iloc[0]), caption=rec.iloc[0], width=150)
                        with col2:
                            st.image(get_poster_url(ids.iloc[1]), caption=rec.iloc[1], width=150)
                        with col3:
                            st.image(get_poster_url(ids.iloc[2]), caption=rec.iloc[2], width=150)
                        with col4:
                            st.image(get_poster_url(ids.iloc[3]), caption=rec.iloc[3], width=150)
                        with col5:
                            st.image(get_poster_url(ids.iloc[4]), caption=rec.iloc[4], width=150)
                        with col6:
                            st.image(get_poster_url(ids.iloc[5]), caption=rec.iloc[5], width=150)
                        with col7:
                            st.image(get_poster_url(ids.iloc[6]), caption=rec.iloc[6], width=150)
                        with col8:
                            st.image(get_poster_url(ids.iloc[7]), caption=rec.iloc[7], width=150)
                        with col9:
                            st.image(get_poster_url(ids.iloc[8]), caption=rec.iloc[8], width=150)
                        with col10:
                            st.image(get_poster_url(ids.iloc[9]), caption=rec.iloc[9], width=150)


                elif select == 'Hybrid based filtering':
                    movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
                    df_c = df[df['id'].isin(movie_id)]
                    df_c.dropna(inplace=True)
                    df_c.drop_duplicates(subset='title', inplace=True)
                    user = st.selectbox('Please select a movie to get recommendations', options=df_c['title'].tolist())
                    user = st.selectbox('Please select a movie to get recommendations', options=df_c['movieId'].tolist())


                    def convert_int(x):
                        try:
                            return int(x)
                        except:
                            return np.nan


                    #md['id'] = md['id'].apply(convert_int)
                    #md[md['id'].isnull()]
                    #md = md.drop([19730, 29503, 35587])
                    #md['id'] = md['id'].astype('int')
                    #smd = md[md['id'].isin(links_small)]
                    #id_map = pd.read_csv(r'./links_small.csv')[
                       # ['movieId', 'tmdbId']]
                    #id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
                   # id_map.columns = ['movieId', 'id']
                   # id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
                    # id_map = id_map.set_index('tmdbId')
                   # indices_map = id_map.set_index('id')


                elif select == 'Item based Collaborative filtering':
                    movies_id = ratings['movieId'].unique()
                    df.dropna(inplace=True)
                    df.drop_duplicates(subset='title', inplace=True)
                    df_l = df[df['id'].isin(movies_id)]
                    ratings1 = ratings[ratings['movieId'].isin(df['id'])]

                    ids = st.selectbox('Please select a movie id',options=df_l.id.to_list())


                    def create_matrix(df):
                        p = len(df['movieId'].unique())
                        q = len(df['userId'].unique())

                        map_user = dict(zip(np.unique(df["userId"]), list(range(q))))
                        map_movie = dict(zip(np.unique(df["movieId"]), list(range(p))))

                        map_user_i = dict(zip(list(range(q)), np.unique(df["userId"])))
                        map_mov_i = dict(zip(list(range(p)), np.unique(df["movieId"])))

                        user_index = [map_user[i] for i in df['userId']]
                        movie_index = [map_movie[i] for i in df['movieId']]

                        matrix = csr_matrix((df["rating"], (movie_index, user_index)), shape=(p, q))

                        return matrix,map_user,map_movie, map_user_i,map_mov_i


                    matrix, map_user,map_movie, map_user_i,map_mov_i = create_matrix(ratings1)



                    def find_similar_movies(movie_id, matrix, k, metric='cosine', show_distance=False):

                        neighbour_ids = []

                        movie_ind = map_movie[movie_id]
                        movie_vec = matrix[movie_ind]
                        k += 1
                        kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
                        kNN.fit(matrix)
                        movie_vec = movie_vec.reshape(1, -1)
                        neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
                        for i in range(0, k):
                            n = neighbour.item(i)
                            neighbour_ids.append(map_mov_i[n])
                        neighbour_ids.pop(0)
                        return neighbour_ids


                    movie_titles = dict(zip(df_l['id'], df_l['title']))
                    movie_ids = dict(zip(df_l['title'], df_l['id']))

                    similar_ids = find_similar_movies(ids, matrix, k=10)
                    movie_title = movie_titles[ids]
                    l1=[]
                    l2=[]
                    for i in similar_ids:
                        l1.append(movie_titles[i])
                    for i in l1:
                        l2.append(movie_ids[i])
                    m = st.markdown("""
                                                <style>
                                  div.stButton > button:first-child {
                                  background-color: #0099ff;
                                          color:#ffffff;
                                                 }
                                   div.stButton > button:hover {
                             background-color: #00ff00;
                                     color:#ff0000;
                                              }
                                </style>""", unsafe_allow_html=True)

                    submit = st.button('Get collaborative filtered recommendations')
                    if submit:
                        st.write(f"Since you watched {movie_title}")
                        st.write(f"Following are the top ten recommendations for you")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col6, col7, col8, col9, col10 = st.columns(5)
                        with col1:
                            st.image(get_poster_url(l2[0]),caption=l1[0],width=150)
                        with col2:
                            st.image(get_poster_url(l2[1]),caption=l1[1],width=150)
                        with col3:

                            st.image(get_poster_url(l2[2]),caption=l1[2],width=150)
                        with col4:

                            st.image(get_poster_url(l2[3]),caption=l1[3],width=150)
                        with col5:

                            st.image(get_poster_url(l2[4]),caption=l1[4],width=150)
                        with col6:

                            st.image(get_poster_url(l2[5]),caption=l1[5],width=150)
                        with col7:

                            st.image(get_poster_url(l2[6]),caption=l1[6],width=150)
                        with col8:

                            st.image(get_poster_url(l2[7]),caption=l1[7],width=150)
                        with col9:

                            st.image(get_poster_url(l2[8]),caption=l1[8],width=150)
                        with col10:

                            st.image(get_poster_url(l2[9]),caption=l1[9],width=150)


                elif select=='User based Collaborative filtering':
                    movies_id = ratings['movieId'].unique()
                    df_l = df[df['id'].isin(movies_id)]
                    ratings1 = ratings[ratings['movieId'].isin(df['id'])]
                    ids = st.selectbox('Please select a user id', options=ratings1['userId'].unique())
                    rating_matrix = ratings1.pivot_table(index='userId', columns='movieId', values='rating')
                    rating_matrix = rating_matrix.fillna(0)


                    def sim(user_id, r_matrix, k=10):
                        user = r_matrix[r_matrix.index == user_id]
                        other_users = r_matrix[r_matrix.index != user_id]
                        sim = cosine_similarity(user, other_users)[0].tolist()
                        idx = other_users.index.to_list()
                        idx_sim = dict(zip(idx, sim))
                        idx_sim_sorted = sorted(idx_sim.items(), key=lambda x: x[1])
                        idx_sim_sorted.reverse()
                        top_user_similarities = idx_sim_sorted[:10]
                        users = [i[0] for i in top_user_similarities]
                        return users

                    s = sim(ids,rating_matrix)


                    def recommend_movie(user_index, similar_user_indices, r_matrix, items=10):
                        similar_users = r_matrix[r_matrix.index.isin(similar_user_indices)]
                        similar_users = similar_users.mean(axis=0)
                        similar_df = pd.DataFrame(similar_users, columns=['mean'])
                        user_df = r_matrix[r_matrix.index == user_index]
                        user_df_transposed = user_df.transpose()
                        user_df_transposed.columns = ['rating']
                        user_df_transposed = user_df_transposed[user_df_transposed['rating'] == 0]
                        movies_unseen = user_df_transposed.index.tolist()
                        similar_users_filtered = similar_df[similar_df.index.isin(movies_unseen)]
                        similar_users_ordered = similar_df.sort_values(by=['mean'], ascending=False)

                        top_movies = similar_users_ordered.head(items)
                        top_movie_indices = top_movies.index.tolist()
                        movie_title = df_l[df_l['id'].isin(top_movie_indices)]['title']
                        movie_id = df_l[df_l['id'].isin(top_movie_indices)]['id']

                        return list(zip(movie_title, movie_id))

                    z = recommend_movie(ids,s,rating_matrix)
                    m = st.markdown("""
                                        <style>
                                div.stButton > button:first-child {
                                background-color: #0099ff;
                                 color:#ffffff;
                                           }
                                div.stButton > button:hover {
                                   background-color: #00ff00;
                                       color:#ff0000;
                                               }
                                   </style>""", unsafe_allow_html=True)

                    submit = st.button('Get recommendations')
                    if submit:
                        st.write(f"Following are the top ten recommendations for you")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col6, col7, col8, col9, col10 = st.columns(5)
                        with col1:
                            st.image(get_poster_url(z[0][1]), caption=z[0][0], width=150)
                        with col2:
                            st.image(get_poster_url(z[1][1]), caption=z[1][0], width=150)
                        with col3:
                            st.image(get_poster_url(z[2][1]), caption=z[2][0], width=150)
                        with col4:
                            st.image(get_poster_url(z[3][1]), caption=z[3][0], width=150)
                        with col5:
                            st.image(get_poster_url(z[4][1]), caption=z[4][0], width=150)
                        with col6:
                            st.image(get_poster_url(z[5][1]), caption=z[5][0], width=150)
                        with col7:
                            st.image(get_poster_url(z[6][1]), caption=z[6][0], width=150)
                        with col8:
                            st.image(get_poster_url(z[7][1]), caption=z[7][0], width=150)
                        with col9:
                            st.image(get_poster_url(z[8][1]), caption=z[8][0], width=150)
                        with col10:
                            st.image(get_poster_url(z[9][1]), caption=z[9][0], width=150)



                elif select=='Hybrid recommendations':
# =============================================================================
#                     links = pd.read_csv(r"./links_small.csv")
#                     df = pd.read_csv(r"./movies.csv")
#                     ratings = pd.read_csv(r"./ratings_small.csv")
#                     svd = SVD()
#                     reader = Reader()
#                     data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#                     trainset = data.build_full_trainset()
#                     svd.fit(trainset)
# =============================================================================


                    def convert_int(x):
                        try:
                            return int(x)
                        except:
                            return np.nan


                    id_map = links[['movieId', 'tmdbId']]
                    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
                    id_map.columns = ['movieId', 'id']
                    id_map = id_map.merge(df[['title', 'id']], on='id').set_index('title')
                    indices_map = id_map.set_index('id')

                    movie_id = links[links['tmdbId'].notnull()]['tmdbId'].astype(int)
                    df_c = df[df['id'].isin(movie_id)]
                    df_c.dropna(inplace=True)
                    df_c.drop_duplicates(subset='title', inplace=True)
                    # user = st.selectbox('Please select a movie to get recommendations',options=df_c['title'].tolist())
                    # submit = st.button('Get recommendations based on content')
                    from nltk import WordNetLemmatizer

                    lemma = WordNetLemmatizer()


                    def lemmatize_text(text):
                        return [lemma.lemmatize(text)]


                    df_c['overview'] = df_c.overview.apply(lemmatize_text)
                    df_c['overview'] = df_c['overview'].apply(lambda x: ' '.join(x))
                    tf = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 6), stop_words='english', analyzer='word')
                    tf_idf = tf.fit_transform(df_c['overview'])
                    sigmoid = sigmoid_kernel(tf_idf, tf_idf)
                    indices = pd.Series(df_c['overview'].index, index=df_c['title'])

                    userid =st.selectbox(label='Userid',options=ratings.userId.unique())
                    movie = st.selectbox(label='movie name',options=df_c.title.to_list())


                    def hybrid(userId, title):
                        idx = indices[title]
                        tmdbId = id_map.loc[title]['id']
                        movie_id = id_map.loc[title]['movieId']

                        sim_scores = list(enumerate(sigmoid[int(idx)]))
                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                        sim_scores = sim_scores[1:26]
                        movie_indices = [i[0] for i in sim_scores]

                        movies = df_c.iloc[movie_indices][['title', 'id']]
                        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
                        movies = movies.sort_values('est', ascending=False)
                        return movies.head(10)
                    submit = st.button('Get Recommendations')
                    if submit:
                      h = hybrid(userid,movie)
                      st.write(f"Following are the top ten recommendations for you based on Hybrid technique")
                      col1, col2, col3, col4, col5 = st.columns(5)
                      col6, col7, col8, col9, col10 = st.columns(5)
                      with col1:
                          st.image(get_poster_url(h['id'].iloc[0]), caption=h['title'].iloc[0], width=150)
                      with col2:
                          st.image(get_poster_url(h['id'].iloc[1]), caption=h['title'].iloc[1], width=150)
                      with col3:
                          st.image(get_poster_url(h['id'].iloc[2]), caption=h['title'].iloc[2], width=150)
                      with col4:
                          st.image(get_poster_url(h['id'].iloc[3]), caption=h['title'].iloc[3], width=150)
                      with col5:
                          st.image(get_poster_url(h['id'].iloc[4]), caption=h['title'].iloc[4], width=150)
                      with col6:
                          st.image(get_poster_url(h['id'].iloc[5]), caption=h['title'].iloc[5], width=150)
                      with col7:
                          st.image(get_poster_url(h['id'].iloc[6]), caption=h['title'].iloc[6], width=150)
                      with col8:
                          st.image(get_poster_url(h['id'].iloc[7]), caption=h['title'].iloc[7], width=150)
                      with col9:
                          st.image(get_poster_url(h['id'].iloc[8]), caption=h['title'].iloc[8], width=150)
                      with col10:
                          st.image(get_poster_url(h['id'].iloc[9]), caption=h['title'].iloc[9], width=150)


                
                
                

            if  menu_Pre_Exp == "Dataset Info" : #and selected == "Projects"
                    st.title('Dataset Info')
                    z = st.radio(label="", options=('Movies data info', 'Rating data info'))
                    if z == 'Movies data info':
                        st.write(' First 5 rows of movies dataset')
                        st.dataframe(df.head())

                        st.write('Last 5 rows of movies dataset')
                        st.dataframe(df.tail())

                        st.write('Description of movies dataset')
                        st.dataframe(df.describe())

                        st.write('shape of our movies dataset')
                        st.write('The number of rows in the movies dataset are')
                        st.write(df.shape[0])
                        st.write('The number of columns in the movies dataset are')
                        st.write(df.shape[1])

                        st.write(' the columns of movies dataset')
                        st.write(df.columns)

                        st.write('the info of movies dataset')
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        s = buffer.getvalue()

                        st.text(s)

                        #st.write('You can find the link for the movies dataset  https://www.kaggle.com/rounakbanik/movie-recommender-systems/data')

                    elif z == 'Rating data info':
                        st.write('First 5 rows of movies_ratings dataset')
                        st.dataframe(ratings.head())

                        st.write('Last 5 rows of movies_ratings dataset')
                        st.dataframe(ratings.tail())

                        st.write('Description of movies_ratings dataset')
                        st.dataframe(ratings.describe())

                        st.write('shape of our movies_ratings dataset')
                        st.write('The number of rows in the movies_ratings dataset are')
                        st.write(ratings.shape[0])
                        st.write('The number of columns in the movies_ratings dataset are')
                        st.write(ratings.shape[1])

                        st.write('The columns of movies_ratings dataset are ')
                        st.write(ratings.columns)

                        st.write('info of movies_ratings dataset')
                        buffer = io.StringIO()
                        ratings.info(buf=buffer)
                        s = buffer.getvalue()

                        st.text(s)

                              
                                                 
                        st.success('Done!')       
                        
                        
                            
                                

                                                      
if __name__=='__main__':
    main()            
            
            





