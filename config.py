import streamlit as st
from PIL import Image
import pandas as pd
from numpy import NaN
import time as Time
from plotly.subplots import make_subplots
import seaborn as sns
import datetime as dt
from scipy import stats
import os.path
import altair as alt
import pickle
from sklearn.metrics import recall_score
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from io import StringIO
import boto3
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,\
                             ConfusionMatrixDisplay, f1_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
import joblib
import datetime
import base64
import warnings
warnings.filterwarnings("ignore")
# Define custom CSS for background images
def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-position: center
         }}
         </style>
         """,
         unsafe_allow_html=True)
    st.markdown(
    """
     <style>
     .st-cb input[type="range"]::-webkit-slider-thumb {
            background-color: blue;
            }
        </style>
    """,
    unsafe_allow_html=True
    )
