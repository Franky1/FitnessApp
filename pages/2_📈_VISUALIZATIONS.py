import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import *
st.set_page_config(page_title = "Training Activities", page_icon = Image.open('Photos/icon.png'), layout="wide")
from func2 import*

def main2():
    s3 = boto3.client('s3', aws_access_key_id = os.environ.get('KEY_ID')
                      aws_secret_access_key = os.environ.get('SECRET_KEY')
    bucket_name = 'fitnessappdata'
    response = s3.get_object(Bucket='fitnessappdata', Key="UserNames.csv")
    csv_contents = response['Body'].read().decode('utf-8')
    existing_users = pd.read_csv(StringIO(csv_contents), index_col = 0)

    set_bg_hack('Photos/backg.png')
    st.write('-'*60)
    expl = """THIS SECTION SHOWS RELEVANT INSIGHTS FROM ALL RECORDED ACTIVITIES.
    GRAPHICS ABOUT THE EVOLUTION OF THE PERFORMANCE, MAXIMUM AND MINIMUM PEAK
    OF PHYSICAL CONDITION, CORRELATIONS BETWEEN FEATURES AND YOUR PERFORMANCE,
    CLUSTERING OF TRAINING PERIODS, ETC."""
    st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#FAFAFA; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl}</p>', unsafe_allow_html=True)
    st.write('-'*60)
    with st.sidebar:
        logout = st.button("Log out")
        if logout:
            for key in st.session_state.keys():
                del st.session_state[key]
    if 'username' not in st.session_state or st.session_state.username == None:
        with st.columns(3)[0]:
            st.error("YOU NEED TO LOG IN FOR USING THE APPLICATION")
    else:
        with st.sidebar:
            st.markdown(f'<p style="text-align: left;color:#006400; font-size:15px; border-radius:2%;">Successfully Log in as {st.session_state.username}</p>', unsafe_allow_html=True)



        ##########################################################################
        ########################### READ AND CLEAN THE DATA ####################
        #########################################################################

        # Read the data and clean it for the visualitzations:
        try:
            data, weekly = read(st.session_state.username, s3)
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
                perf = """THE PERFORMANCE SCORE IS A NUMERICAL VALUE FROM 1 TO 10
                    THAT THE SYSTEM ASSIGNS TO EACH OF THE ACTIVITIES BASED ON THE RECORDED
                    DATA. AS THIS NUMBER INCREASES, IT INDICATES SUPERIOR PERFORMANCE,
                    IMPLYING THAT THE TRAINING RESULTS ARE SUPERIOR COMPARED TO SIMILAR WORKOUTS."""
                for i in range(3):
                    st.markdown("")
                st.markdown(f'<p style="text-align: left; padding: 20px; background-color:#FAFAFA; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{perf}</p>', unsafe_allow_html=True)



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
            with st.expander("EXTRA INFORMATION:"):
                count = """
                THESE ARE A BAR AND A CLUSTERED BAR CHARTS THAT SHOW THE NUMBER OF
                ACTIVITIES FOR EACH ACTIVITY TYPE AND MONTH. THIS VISUALIZATION MAY HELP YOU
                IDENTIFY PATTERNS IN THE DATA, SUCH AS YOUR PREFERRED ACTIVITY TYPES.
                """
                st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{count}</p>', unsafe_allow_html=True)



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
            with st.expander("EXTRA INFORMATION:"):
                numerical = """
                THESE ARE A BAR AND A CLUSTERED BAR CHART THAT SHOW THE TOTAL AMOUNT OF A SPECIFIED NUMERICAL FEATURE FOR
                EACH ACTIVITY TYPE AND MONTH. FOR INSTANCE, THIS VISUALIZATION ALLOW YOU TO IDENTIFY WHICH ACTIVITY TYPES YOU
                HAVE SPENT MORE TIME ON (IF THE NUMERICAL FEATURE IS TIME), OR WHICH ACTIVITY TYPES ALLOW YOU TO COVER MORE DISTANCE
                (IF THE NUMERICAL FEATURE IS DISTANCE).
                """
                st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{numerical}</p>', unsafe_allow_html=True)



            #########################################################################
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
            with st.expander("EXTRA INFORMATION:"):
                corr = """
                THESE ARE A SCATTERPLOT AND A HEATMAP. THE FIRST ONE SHOWS THE CORRELATION BETWEEN TWO NUMERICAL FEATURES SPECIFIED BY YOU,
                FILTERED BY ACTIVITY TYPE. THE SECOND ONE SHOWS THE AMOUNT OF A NUMERICAL FEATURE FOR EACH ACTIVITY TYPE AND MONTH. THESE VISUALIZATIONS HELP
                YOU UNDERSTAND AND DISCOVER CORRELATIONS BETWEEN NUMERICAL FEATURES.
                """
                st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{corr}</p>', unsafe_allow_html=True)

            ##########################################################################
            ############################### DISTRIBUTION #############################
            ##########################################################################

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
            with st.expander("EXTRA INFORMATION:"):
                distr = """
                THIS VISUALIZATION INCLUDES A BOX PLOT THAT SHOWS THE MEAN AND STANDARD DEVIATION FOR A SPECIFIED NUMERICAL FEATURE, AND A HISTOGRAM THAT SHOWS
                THE DISTRIBUTION OF THE FEATURE, FILTERED BY ACTIVITY TYPE. THIS HELPS THE USER UNDERSTAND THE DISTRIBUTION OF NUMERICAL FEATURES AND IDENTIFY ANY OUTLIERS
                OR RARE VALUES.
                """
                st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{distr}</p>', unsafe_allow_html=True)

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
