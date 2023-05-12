#import streamlit as st
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import *
st.set_page_config(page_title = "Training Activities", layout="wide", page_icon = Image.open('Photos/icon.png'))
from func2 import*

def main2():
    s3 = boto3.client('s3', aws_access_key_id='AKIAW4BRQNULH32VEHOM',
                      aws_secret_access_key='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')
    bucket_name = 'fitnessappdata'
    response = s3.get_object(Bucket='fitnessappdata', Key="UserNames.csv")
    csv_contents = response['Body'].read().decode('utf-8')
    existing_users = pd.read_csv(StringIO(csv_contents), index_col = 0)

    set_bg_hack('Photos/backg.png')

    st.write('-'*60)

    expl = """THIS SECTION SHOWS RELEVANT INSIGHTS FROM ALL RECORDED ACTIVITIES.
    GRAPHICS ABOUT THE EVOLUTION OF THE PERFORMANCE, MAXIMUM AND MINIMUM PEAK
    OF PHYSICAL CONDITION, CORRELATIONS BETWEEN FEATURES AND USER'S PERFORMANCE,
    CLUSTERING OF TRAINING PERIODS, ETC."""
    st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#FAFAFA; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl}</p>', unsafe_allow_html=True)
    with st.columns((2, 3.3))[0]:
        username = st.text_input("WRITE YOUR USER NAME", "", help = "Write your User Name")
        if not existing_users["UserName"].str.contains(username).any():
            st.error(f"{username} doesn't exist")
    st.write('-'*60)


    ##########################################################################
    ########################### READ AND CLEAN THE DATA ####################
    #########################################################################

    # Read the data and clean it for the visualitzations:
    try:
        data, weekly = read(username, s3)
        data = clean3(data)

        # Add filter for all visualitzations
        st.sidebar.markdown("<h1 style='text-align: center; font-size:20px; '>FILTER FOR ALL PLOTS</h1>", unsafe_allow_html=True)
        start_date = st.sidebar.date_input('START DATE', data['Date'].min())
        end_date = st.sidebar.date_input('END DATE', data['Date'].max())
        data = data[(data['Date'].dt.date > start_date) & (data['Date'].dt.date < end_date)]

        ##########################################################################
        ###################### EVOLIUTIONS OF THE PERFORMANCE ####################
        #########################################################################

        title("PERFORMANCE EVOLUTION")
        chart, MID, expl = st.columns((2, 1, 2))
        with chart:
                #with st.columns((1, 6, 0.1))[1]:
            year = st.radio('Select a year',list(data['Date'].dt.year.unique()), help = "Select a year for the following chart", label_visibility = "collapsed")
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            chart1 = perf_evol(data, year)
            st.altair_chart(chart1)
        with expl:
            df = data[data['Year'] == year].groupby('MonthName').mean()
            max = np.max(df.Performance)
            date_max = df[df['Performance'] == max].index
            min = np.min(df.Performance)
            date_min = df[df['Performance'] == min].index
            c1, mid, c2 = st.columns([2.8, 1, 3])
            with c1:
                res_max = date_max[0].upper() + ": " + str(round(max, 2))
                st.markdown("<p style = font-size:20px; > MAX. PERFORMANCE: ", unsafe_allow_html = True)
                st.markdown(f'<p style = color:#119A05; font-size:10px;">{res_max}</p>', unsafe_allow_html=True)
            with mid:
                image = Image.open('Photos/line.png').resize((300, 200))
                st.write('')
                st.image(image)
            with c2:
                res_min = date_min[0].upper() + ": " + str(round(min, 2))
                st.markdown("<p style = font-size:20px; > MIN. PERFORMANCE: ", unsafe_allow_html = True)
                st.markdown(f'<p style = color:#F93419; font-size:10px;">{res_min}</p>', unsafe_allow_html=True)
            st.code("EXPLANATION OF THE CHART AND PERF CALCULATION")


        ##########################################################################
        ########################### COUNT OF ACTIVITIES #########################
        #########################################################################


        st.write('')
        st.write('')
        title("ACTIVITIES COUNT:")
        ch1, ch2 = st.columns((0.6, 2))
        with ch1:
            st.markdown(f'<p style="text-align: center; color:#474747; font-size:20px;">TOTAL COUNT</p>', unsafe_allow_html=True)
            st.write('')
            st.write('')
            st.write('')
            chart2 = count_activities(data)
            st.altair_chart(chart2)
        with ch2:
            st.markdown(f'<p style="text-align: center; color:#474747; font-size:20px;">COUNT PER MONTH</p>', unsafe_allow_html=True)
            act_types = st.multiselect('Select activity types', list(data['ActivityType'].unique()), default = list(data['ActivityType'].unique()), label_visibility = "collapsed")
            chart3 = count_act_month(data[data['ActivityType'].isin(act_types)])
            st.altair_chart(chart3)


        ##########################################################################
        ######################## NUMERICAL X ACTIVITY TYPE ######################
        #########################################################################


        title("SUMMARY OF NUMERICAL FEATURES")
        with st.columns((1,1,2))[0]:
            var = st.selectbox('Select numerical variable:', data.select_dtypes(include=np.number).columns.tolist(), label_visibility = "collapsed")
        st.write('')
        ch1, ch2 = st.columns((0.6, 2))
        with ch1:
            st.markdown(f'<p style="text-align: center; color:#474747; font-size:20px;">TOTAL SUM</p>', unsafe_allow_html=True)
            chart5 = bar(data, var)
            st.altair_chart(chart5)
        with ch2:
            st.markdown(f'<p style="text-align: center; color:#474747; font-size:20px;">SUM PER MONTH</p>', unsafe_allow_html=True)
            chart51 = count_num_month(data, var)
            st.altair_chart(chart51)


        ##########################################################################
        ############################### CORRELATION #############################
        #########################################################################

        title("CORRELATION ANALYSIS")
        sc, mid, hm = st.columns((3, 0.5, 1.8))
        with sc:
            v1, v2 = st.columns((1, 1))
            act_types2 = st.multiselect('Select activity types: ', list(data['ActivityType'].unique()), default = list(data['ActivityType'].unique()), label_visibility = "collapsed")
            with v1:
                var1 = st.selectbox('Select first variable:', data.select_dtypes(include=np.number).columns.tolist(), label_visibility = "collapsed")
            with v2:
                var2 = st.selectbox('Select second variable:', data.select_dtypes(include=np.number).columns.tolist(), label_visibility = "collapsed")
            chart4 = scater(data[data['ActivityType'].isin(act_types2)], var1, var2, act_types)
            st.altair_chart(chart4)
        with hm:
            var3 = st.selectbox('Select num variable:', data.select_dtypes(include=np.number).columns.tolist(), label_visibility = "collapsed")
            chart6 = heat(data, var3)
            st.altair_chart(chart6)

        title("DISTRIBUTION ANALYSIS")
        bx, mid, hist = st.columns((1, 0.2, 2))
        with bx:
            var4 = st.selectbox('Select num variables:', data.select_dtypes(include=np.number).columns.tolist(), label_visibility = "collapsed")
            for i in range(5):
                st.write('')
            chart7 = box(data, var4)
            st.altair_chart(chart7)
        with hist:
            var5 = st.selectbox('Selec num variables:', data.select_dtypes(include=np.number).columns.tolist(),label_visibility = "collapsed")
            act_types3 = st.multiselect('Select activity type: ', list(data['ActivityType'].unique()), default =  "Road biking", label_visibility = "collapsed")
            chart8 = histogram(data[data['ActivityType'].isin(act_types3)], var5)
            st.altair_chart(chart8)
    except:
        no_req = "THERE IS NO DATA SAVED. MAKE SURE TO UPLOAD YOUR DATA IN THE FIRST SECTION."
        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_req}</p>', unsafe_allow_html=True)


footnote_css("style.css")
with st.container():
    ima1, mid, ima2 = st.columns((1,4,1))
    with ima1:
        image = Image.open('Photos/logo.png').resize((200, 50))
        st.image(image)
    with mid:
        st.markdown("<h1 style='text-align: center; font-size:40px; color: #414345;'>TRAINING ACTIVITIES DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size:25px; color: #414345;'>VISUALITZATION OF THE DATA</h1>", unsafe_allow_html=True)
    with ima2:
        for i in range(5):
            st.write('')
        image = Image.open('Photos/logoUT.png').resize((180, 60))
        st.image(image)
for i in range(4):
    st.write('')
main2()
for i in range(4):
    st.write('')
ima1, mid, ima2 = st.columns((1,4,1))
with ima1:
    image = Image.open('Photos/logo.png').resize((200, 50))
    st.image(image)
with ima2:
    image = Image.open('Photos/logoUT.png').resize((180, 60))
    st.image(image)
