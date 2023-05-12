from config import*
st.set_page_config(page_title = "Training Activities", layout="wide", page_icon = Image.open('Photos/icon.png'))
from func_main import *


def main():
    s3 = boto3.client('s3', aws_access_key_id='AKIAW4BRQNULH32VEHOM',
                      aws_secret_access_key='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')
    bucket_name = 'fitnessappdata'
    set_bg_hack('Photos/backg.png')
    st.write('-'*60)
    expl = """THIS SECTION AIMS TO IDENTIFY THE MOST OPTIMAL MODEL THAT CAN
    ACCURATELY PREDICT THE PERFORMANCE FOR THE TRAININGS OF THE UPCOMING WEEK.
    THIS IS ACHIEVED BY EITHER USING THE EXISTING DATA THAT HAS BEEN
    SAVED, OR BY USING A COMPLETELY NEW DATASET. ONCE THE USER HAS REVIEWED AND
    EVALUATED THE OUTPUT OF THE MODEL, THEY CAN PROCEED TO SAVE IT BY CLICKING ON THE "SAVE" BUTTON.
    THIS WILL ENABLE THEM TO USE THE MODEL IN THE FUTURE."""
    st.markdown(f'<p style="text-align: center; padding: 10px; background-color:#FAFAFA; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl}</p>', unsafe_allow_html=True)
    st.write('-'*60)

            ########################################################################
            ######################### READ AND CLEAN THE DATA ######################
            ########################################################################


    st.markdown(f'<p style="text-align: center; color:#303030; font-size:40px;">READ AND CLEAN THE DATA</p>', unsafe_allow_html=True)
    sel, oth, other = st.columns((2,0.3,3))
    user = st.radio("new/old",('NEW USER', 'EXISTENT USER'), label_visibility = 'collapsed', horizontal = True)
    response = s3.get_object(Bucket='fitnessappdata', Key="UserNames.csv")
    csv_contents = response['Body'].read().decode('utf-8')
    existing_users = pd.read_csv(StringIO(csv_contents), index_col = 0)
    if user == "NEW USER":
        # Ask for the user name
        with st.columns((2, 3.3))[0]:
            username = st.text_input("WRITE A NEW USER NAME", "", help = "Write a User Name")
            # Check it doesn't exist
            if existing_users["UserName"].str.contains(username).any() and username != "":
                st.error(f"{username} already exists.")
            else:
                if username != "":
                    existing_users = existing_users.append({'UserName':username}, ignore_index = True)
                    # Save the user name
                    csv_buffer = StringIO()
                    existing_users.to_csv(csv_buffer)
                    csv_data = csv_buffer.getvalue().encode('utf-8')
                    s3.upload_fileobj(BytesIO(csv_data), 'fitnessappdata', "UserNames.csv")
                    st.markdown(f'<p style="text-align: left; padding: 10px; color:#006400; font-size:15px; border-radius:2%;">UserName correctly stored</p>', unsafe_allow_html=True)
    else:
        with st.columns((2, 3.3))[0]:
            username = st.text_input("WRITE YOUR USER NAME", "", help = "Write your User Name")
            if not existing_users["UserName"].str.contains(username).any():
                st.error(f"{username} doesn't exist")

    st.session_state.username = username
    sb, mid, expl = st.columns((2,0.3,3))
    with sb:
        opt = st.selectbox('NEW DATA OR SAVED DATA??', ('SELECT ONE OPTION', 'NEW DATA', 'SAVED DATA'), label_visibility="collapsed")
    with expl:
        expl2 = """At this stage, you have the option to either use completely new data or previously saved data:"""
        expl21 = """=> New data is particularly relevant for new users who
        have not yet saved any data. Note that this option will result in the loss of any
        previously saved data and will cause the entire dashboard to adjust and reflect the new dataset,
        effectively replacing any previously saved information."""
        expl22 = """=> Saved data means that the app will only include new activities that
        were saved since the last time the model was trained, and then retrain the model."""

        with st.expander("EXTRA INFORMATION:"):
            st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl2}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl21}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{expl22}</p>', unsafe_allow_html=True)
    if opt == 'NEW DATA':
        ## LET THE USER UPLOAD THEIR DATA
        col1, mid, col2 = st.columns((2,0.3, 3))
        with col1:
            uploadedFile = st.file_uploader('uploadedFile', type=['csv','xlsx'],accept_multiple_files=False,key="uploadedFile", label_visibility = "collapsed")
        with col2:
            with open("Data/RequirementsData.docx", 'rb') as f:
                data = f.read()
            if st.button(':page_facing_up: Download Requirements Data File'):
                tmp_download_link = create_download_link(data, "Data/RequirementsData.docx")
                st.markdown(tmp_download_link, unsafe_allow_html=True)
        if uploadedFile is not None:
            try:
                data = read(uploadedFile)
                data = clean(data)
                data = remove_outliers_quartiles(data)
                with st.columns((1,2))[0]:
                    ok = "DATA READ CORRECTLY"
                    st.markdown(f'<p style="text-align: left; color:#006400; font-size:20px; border-radius:2%;">{ok}</p>', unsafe_allow_html=True)
            except:
                error = "THE DATA COULD NOT BE READ CORRECTLY. MAKE SURE THAT THE DATABASE MEETS THE SPECIFIED REQUIREMENTS."
                st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{error}</p>', unsafe_allow_html=True)
                return

            ## LABEL THE DATASET (PERFORMANCE CALCULATION + GROUP BY WEEKS)
            feat = data.columns
            weekly, data = group_by_weeks(data)
            data, weekly = perf_label(data[feat], weekly)

            ## CHECK IF THE DATASETS ARE BALANCED, AND IF NOT BALANCE IT
            bal = check(weekly)
            if not bal:
                my_dict = dict(weekly['Label'].value_counts())
                not_bal = """The labeled database that has been generated is imbalanced,
                meaning that it does not have an equal number of registers for Maintenance, Positive and Negative weeks.
                Specifically, it has """ + str(my_dict['Maintenance']) + """ Maintenance weeks,
                """ + str(my_dict['Positive']) + """ Positive weeks and """ + str(my_dict['Negative']) + """ Negative weeks.
                To improve the accuracy of our model, the app will generate synthetic data that mainly follows the distribution of the two minority classes."""
                st.markdown(f'<p style="text-align: left; padding: 10px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{not_bal}</p>', unsafe_allow_html=True)
            while not bal:
                bal, data, weekly = balance(weekly, data, feat)
            with st.columns((1,2))[0]:
                my_dict2 = dict(weekly['Label'].value_counts())
                ok_bal = "WEEKLY DATASET BALANCED: "
                ok_bal2 = str(my_dict2['Maintenance']) + """ Maintenance weeks, """ + str(my_dict2['Positive']) + """ Positive weeks and """ + str(my_dict2['Negative']) + """ Negative weeks."""
                st.markdown(f'<p style="text-align: left;color:#006400; font-size:20px; border-radius:2%;">{ok_bal}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="text-align: left;color:#006400; font-size:15px; border-radius:2%;">{ok_bal2}</p>', unsafe_allow_html=True)


            # Save both weekly and original datasets in AWS
            object_key1 = f"Data/{username}/Data.csv"
            object_key2 = f"Data/{username}/Weekly.csv"

            csv_buffer1 = StringIO()
            data.to_csv(csv_buffer1, index=False)
            s3.put_object(Bucket=bucket_name, Key=object_key1, Body=csv_buffer1.getvalue())
            st.write("hola")

            csv_buffer2 = StringIO()
            weekly.to_csv(csv_buffer2, index = True)
            s3.put_object(Bucket=bucket_name, Key=object_key2, Body=csv_buffer2.getvalue())

            # Allow the user to download it:
            dt, we, pad = st.columns((1,1,3))
            with dt:
                st.download_button(
                label=":page_facing_up: Cleaned data as CSV",
                help = "Press to download cleaned data as CSV",
                data=convert_df(data),
                file_name='data.csv',
                mime='text/csv',
                )
            with we:
                st.download_button(
                label=":page_facing_up: Weekly data as CSV",
                help = "Press to download weekly data as CSV",
                data=convert_df(weekly),
                file_name='weekly.csv',
                mime='text/csv',
                )


    elif opt == 'SAVED DATA':
        try:
            #FROM AWS:
            s3 = boto3.client('s3', aws_access_key_id='AKIAW4BRQNULH32VEHOM',
                      aws_secret_access_key='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')
            k1 = f"Data/{username}/Data.csv"
            k2 = f"Data/{username}/Weekly.csv"

            response1 = s3.get_object(Bucket='fitnessappdata', Key=k1)
            csv_contents1 = response1['Body'].read().decode('utf-8')
            response2 = s3.get_object(Bucket='fitnessappdata', Key=k2)
            csv_contents2 = response2['Body'].read().decode('utf-8')

            # Convert the CSV data into a Pandas DataFrame
            data = pd.read_csv(StringIO(csv_contents1), index_col = 0)
            weekly = pd.read_csv(StringIO(csv_contents2), index_col = 0).rename(columns={'Week.1': 'Week'})
            last_modified_date = response1['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"The date of the last modification of the data is {last_modified_date}")

        except:
            error = "THERE IS NO DATA SAVED. IF YOU ARE A NEW USER YOU MUST UPLOAD YOUR OWN DATA THE FIRST TIME."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{error}</p>', unsafe_allow_html=True)


    ############################################################################
    ######################### TRAIN THE MODEL #################################
    ###########################################################################


    st.write('-'*60)
    st.markdown(f'<p style="text-align: center; color:#303030; font-size:40px;">TRAIN THE MODEL</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns((2, 2, 2))
    with c1:
        st.markdown(f'<p style="text-align: center; color:#303030; font-size:20px;">MINIMUM ACCURACY</p>', unsafe_allow_html=True)
        acc = st.slider("Minimum accuracy", min_value=0.0, max_value=1.0, value = 0.60, label_visibility = "collapsed")
    with c2:
        st.markdown(f'<p style="text-align: center; color:#303030; font-size:20px;">MINIMUM RECALL</p>', unsafe_allow_html=True)
        rec = st.slider("Minimum recall", min_value=0.0, max_value=1.0, value = 0.60, label_visibility = "collapsed")
    with c3:
        st.markdown(f'<p style="text-align: center; color:#303030; font-size:20px;">MAXIMUM TIME (MINUTES)</p>', unsafe_allow_html=True)
        time = st.slider("Maximum time", min_value=0, max_value=120, value = 5, label_visibility = "collapsed", key = 'sl')
    with st.expander("EXTRA INFORMATION: "):
        exp_metrics1 = "Accuracy and recall are two commonly used metrics in machine learning and data analysis that are used to evaluate the performance of a model or algorithm."
        exp_metrics2 = "=> Accuracy is a measure of how well a model is able to correctly classify examples. It is defined as the percentage of correct predictions made by the model over all predictions made."
        exp_metrics3 = "=> Recall is a measure of how well a model is able to correctly identify examples of one specific class. It is defined as the percentage of correctly identified examples over all examples that are from an specific."
        exp_metrics4 = "=> Also, there is the chance of choosing the time duration during which the app will search for a model that meets the specified conditions. If the app is unable to find a suitable model within the specified time limit, the search will be terminated and you will be notified that it was not possible to find a model that satisfies the given conditions"
        st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{exp_metrics1}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{exp_metrics2}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{exp_metrics3}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: left; padding: 20px; color:#303030; font-size:18px; border-color:#EEEEEE; borderwidth:20px; border-radius:2%;">{exp_metrics4}</p>', unsafe_allow_html=True)

    if st.button(":arrow_forward: RUN", help = "RUN TO FIND A SUITABLE MODEL"):
        try:
            ret, df, model, X_train, X_test, y_train, y_test, keylist, y_pred, found = find_model(weekly, acc, rec, time)
            if not found:
                timeout = "TIMEOUT. IT WAS NOT POSSIBLE TO FIND A MODEL WITH THE SPECIFIED CONDITIONS. TRY TO DECREASE THE MINIMUM METRICS."
                st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{timeout}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="text-align: center; color:#303030; font-size:30px;">METRICS OF THE MODEL</p>', unsafe_allow_html=True)
                res1, res2, cm = metrics(weekly, model, X_train, X_test, y_train, y_test, keylist, y_pred)
                g, m, c = st.columns((2.5,0.1,3))
                with c:
                    for i in range(4):
                        st.write('')
                    st.altair_chart(cm)
                with g:
                    st.markdown(f'<p style="text-align: left; color:#303030; font-size:20px;">GENERAL METRICS:</p>', unsafe_allow_html=True)
                    st.dataframe(res2)
                    st.markdown(f'<p style="text-align: left; color:#303030; font-size:20px;">METRICS FOR EACH CLASS:</p>', unsafe_allow_html=True)
                    st.dataframe(res1.head(3))
                st.session_state.model = model
                st.session_state.df = df
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.keylist = keylist
                st.session_state.y_pred = y_pred
        except:
            error = "THERE IS NO DATA SAVED. MAKE SURE THAT YOU HAVE UPLOADED THE DATA."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{error}</p>', unsafe_allow_html=True)


    ############################################################################
    ######################### SAVE THE MODEL #################################
    ###########################################################################

    if st.button(":floppy_disk:", help = "SAVE"):
        try:
            save(st.session_state.model, st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.keylist, st.session_state.y_pred, weekly, data, s3, bucket_name, username)
            st.write("Model saved :heavy_check_mark:")
        except:
            error = "THERE IS NOT ANY MODEL TO SAVE. MAKE SURE THAT THE APPLICATION HAS FOUND A MODEL."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{error}</p>', unsafe_allow_html=True)


    ############################################################################
    ######################### SEE SAVED MODEL #################################
    ###########################################################################

    with st.expander("SEE THE CURRENTLY SAVED MODEL: "):
        try:
            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/Model.sav")
            sav_file = response['Body'].read()
            model = pickle.loads(sav_file)

            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/key.txt")
            lines = response['Body'].read().decode('utf-8').split('\n')
            lines = list(filter(None, lines))
            keylist = [line for line in lines]


            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/X_train.npy")
            X_train = pickle.loads(response['Body'].read())

            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/X_test.npy")
            X_test = pickle.loads(response['Body'].read())

            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/y_train.npy")
            y_train = pickle.loads(response['Body'].read())

            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/y_test.npy")
            y_test = pickle.loads(response['Body'].read())

            response = s3.get_object(Bucket=bucket_name, Key=f"Model/{username}/y_pred.npy")
            y_pred = pickle.loads(response['Body'].read())


            response = s3.get_object(Bucket='fitnessappdata', Key=f"Data/{username}/Weekly.csv")
            csv_contents = response['Body'].read().decode('utf-8')
            weekly = pd.read_csv(StringIO(csv_contents), index_col = 0).rename(columns={'Week.1': 'Week'})


            res1, res2, cm = metrics(weekly, model, X_train, X_test, y_train, y_test, keylist, y_pred)
            g, m, c = st.columns((2.5,0.1,3))
            with c:
                for i in range(4):
                    st.write('')
                st.altair_chart(cm)
            with g:
                st.markdown(f'<p style="text-align: left; color:#303030; font-size:20px;">GENERAL METRICS:</p>', unsafe_allow_html=True)
                st.dataframe(res2)
                st.markdown(f'<p style="text-align: left; color:#303030; font-size:20px;">METRICS FOR EACH CLASS:</p>', unsafe_allow_html=True)
                st.dataframe(res1.head(3))
        except:
            no_mod = "THERE IS NOT ANY SAVED MODEL."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_mod}</p>', unsafe_allow_html=True)


footnote_css("style.css")
with st.container():
    ima1, mid, ima2 = st.columns((1,4,1))
    with ima1:
        image = Image.open('Photos/logo.png').resize((200, 50))
        st.image(image)
    with mid:
        st.markdown("<h1 style='text-align: center; font-size:40px; color: #414345;'>TRAINING ACTIVITIES DASHBOARD</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size:25px; color: #414345;'>TRAINING THE MODEL</h1>", unsafe_allow_html=True)
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
