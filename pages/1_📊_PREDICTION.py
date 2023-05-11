# -*- coding: utf-8 -*-
"""TFG_Main_Script.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Q2i9seqVoAc1jSJ7wi08qCV1qb-bEUo-
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from config import *
st.set_page_config(page_title = "Training Activities", layout="wide", page_icon = Image.open('Photos/icon.png'))
from func1 import*




def main():
    s3 = boto3.client('s3', aws_access_key_id='AKIAW4BRQNULH32VEHOM',
                      aws_secret_access_key='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')
    bucket_name = 'fitnessappdata'
    response = s3.get_object(Bucket='fitnessappdata', Key="UserNames.csv")
    csv_contents = response['Body'].read().decode('utf-8')
    existing_users = pd.read_csv(StringIO(csv_contents), index_col = 0)

    # Define the tite:
    set_bg_hack('Photos/backg.png')
    st.write('-'*60)
    #st.markdown("<h1 style='text-align: center; font-size:50px; color: black;'>TRAINING ACTIVITIES DASHBOARD</h1>", unsafe_allow_html=True)
    expl = """THIS SECTION PRESENTS THE USER'S FITNESS PERCENTAGE, WHICH IS
    BASED ON THEIR LAST WEEK'S TRAINING COMPARED TO THEIR OVERALL FITNESS HISTORY.
    ESSENTIALLY, THIS COMPARISON EVALUATES THE USER'S PRESENT PHYSICAL CONDITION AGAINST
    THEIR PERSONAL HIGHEST AND LOWEST LEVELS. THE OUTCOME OF THIS ANALYSIS INFORMS THE USER
    IF THEIR LAST WEEK'S TRAINING HAS BEEN POSITIVE, NEGATIVE, OR MAINTENANCE, IN TERMS OF
    THEIR FUTURE PERFORMANCE. FURTHERMORE, THE SYSTEM PROVIDES A HELPFUL SUGGESTION MESSAGE TO THE USER."""
    st.markdown(f'<p style="text-align: center; padding: 10px; background-color:#FAFAFA; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl}</p>', unsafe_allow_html=True)
    st.write('-'*60)
    with st.columns((2, 3.3))[0]:
            username = st.text_input("WRITE YOUR USER NAME", "", help = "Write your User Name")
            if not existing_users["UserName"].str.contains(username).any():
                st.error(f"{username} doesn't exist")

        ############################################################################
        ##################### PREDICTION LABEL FOR NEW ACTIVITIES #################
        ###########################################################################

    st.markdown("<h1 style='text-align: center; font-size:40px; color: black;'>NEW WEEKS</h1>", unsafe_allow_html=True)
    info = "MODIFY THE FOLLOWING DATASET ACCORDING TO THE ACTIVITIES YOU HAVE DONE DURING THE LAST WEEK. THEN PRESS THE BUTTON."
    st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#FAFAFA; color:#303030; font-size:18px; border-radius:2%;">{info}</p>', unsafe_allow_html=True)


    df = pd.DataFrame(
        {"ActivityType": 'Select Activity type', "Date": 'yyyy-mm-dd', \
        "Title": '--', "Distance": 'In Km', "Calories": '--', "Time": 'In minutes',\
        "AverageHeartRate": '--', "MaximumHeartRate": '--', "AverageSpeed": 'In Km', "MaximumSpeed": 'In Km',\
        "TotalAscent": 'In meters', "TotalDescent": 'In meters', "AveragePower": '--', "Difficulty": '--',\
        "MinimumTemperature": 'In celsius degrees', "MaximumTemperature": 'In celsius degrees', "MovingTime": 'In minutes', "ElapsedTime": 'In minutes',\
        "MinimumAltitude": 'In meters', "MaximumAltitude": 'In meters'}, index = [0])
    df['ActivityType'] = (df['ActivityType'].astype("category").cat.add_categories(['Running', 'Road biking', 'Mountain biking', 'Spinning', 'Weight training']))
    new_activities = st.experimental_data_editor(df, num_rows="dynamic")
    try:
        data = pd.read_csv('Data/data.csv', index_col = 0)
        weekly = pd.read_csv('Data/weekly.csv', index_col = 0).rename(columns={"Week.1": "Week"})
    except:
        error = "THERE IS NOT ANY DATA SAVED. MAKE SURE TO UPLOAD YOUR DATA IN THE PREVIOUS SECTION."
        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{error}</p>', unsafe_allow_html=True)


    if st.button('Done', help = "Press the button once you have filled the dataset with the new activities."):
        try:
            new_activities = clean2(new_activities)
            df, new_weeks = perf_label(new_activities, data, weekly)
            new_weeks, result = predict(new_weeks, weekly)

                # Compare the performance with entire history:
            new_weeks = compare(weekly, new_weeks)

                # recommendation messages
            st.markdown(f'<p style="text-align: center; color:#303030; font-size:30px;">SUMMARY OF NEW WEEKS</p>', unsafe_allow_html=True)
            st.write(new_weeks)
            new_weeks = recomendation(new_weeks, weekly, data, result)

                # Print the result: one tab for each week
            show(new_weeks)

            st.session_state.weeks = new_weeks
            st.session_state.act = df
        except:
            no_req = "THE DATA DOES NOT SATISFY THE REQUIREMENTS. MAKE SURE THAT ALL THE VALUES STISFY THE METRICS SPECIFIED AND THAT THERE IS NOT ANY NULL VALUE."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_req}</p>', unsafe_allow_html=True)


        ############################################################################
        ############################## SAVE THE RESULTS ###########################
        ###########################################################################

    if st.button(":floppy_disk:", help = "APPEND THE NEW ACTIVITIES IN THE SAVED DATASET"):
        try:
            save(data, weekly, st.session_state.weeks, st.session_state.act, username, s3, bucket_name)
        except:
            no_req = "THERE IS NOTHING TO SAVE."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_req}</p>', unsafe_allow_html=True)



        ############################################################################
        ############################## SHOW PREVIOUS RESULTS ######################
        ###########################################################################

    # Show results of previous weeks:
    st.markdown("<h1 style='text-align: center; font-size:40px; color: black;'>OLD WEEKS</h1>", unsafe_allow_html=True)
    with st.expander("EXPAND TO SEE PREVIOUS WEEKS"):
            # Read again just in case the user has appended new data:
        try:
            data = pd.read_csv('Data/data.csv', index_col = 0)
            weekly = pd.read_csv('Data/weekly.csv', index_col = 0).rename(columns={"Week.1": "Week"})
            data['Date'] = pd.to_datetime(data['Date'])
            dy = st.date_input('Select any date:', data['Date'].max(),\
                 min_value = data['Date'].min(), max_value = data['Date'].max(), label_visibility="collapsed")
            prev_weeks, w = search_prev(dy, weekly, data)
            # Show the activities of that week:
            txt = 'THE ACTIVITIES OF THE SELECTED WEEK WERE THE FOLLOWING:'
            st.markdown(f'<p style="text-align: center; padding: 20px; color:black; font-size:28px;">{txt}</p>', unsafe_allow_html=True)
            data[(data['Date'].dt.date >= w) & (data['Date'].dt.date <= w + pd.to_timedelta(6 , unit='d'))]

            # Show the results of that week:
            txt = 'THE RESULTS OF THE SELECTED WEEK WERE THE FOLLOWING:'
            st.markdown(f'<p style="text-align: center; padding: 20px; color:black; font-size:28px;">{txt}</p>', unsafe_allow_html=True)
            show(prev_weeks)
        except:
            no_req = "THERE IS NOT ANY DATA SAVED."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_req}</p>', unsafe_allow_html=True)



footnote_css("style.css")
with st.container():
    ima1, mid, ima2 = st.columns((1,4,1))
    with ima1:
        image = Image.open('Photos/logo.png').resize((200, 50))
        st.image(image)
    with mid:
        st.markdown("<h1 style='text-align: center; font-size:40px; color: #414345;'>TRAINING ACTIVITIES DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size:25px; color: #414345;'>PREDICITION FOR NEW WEEKS</h1>", unsafe_allow_html=True)
    with ima2:
        for i in range(5):
            st.write('')
        image = Image.open('Photos/logoUT.png').resize((180, 60))
        st.image(image)
for i in range(4):
    st.write('')
main()
for i in range(4):
    st.write('')
ima1, mid, ima2 = st.columns((1,4,1))
with ima1:
    image = Image.open('Photos/logo.png').resize((200, 50))
    st.image(image)
with ima2:
    image = Image.open('Photos/logoUT.png').resize((180, 60))
    st.image(image)
