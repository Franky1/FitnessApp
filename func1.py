from config import *

#data = pd.read_csv('Data/data.csv', index_col = 0)
#weekly = pd.read_csv('Data/weekly.csv', index_col = 0).rename(columns={"Week.1": "Week"})

def clean2(data):
  '''Function that given the raw dataset from GARMIN APP, cleans it'''

  # From str to datetime
  data['Date'] = pd.to_datetime(data['Date']).dt.date

  #To Floats
  floats = ['Distance', 'Calories', 'AverageHeartRate', 'MaximumHeartRate', \
            'TotalAscent', 'TotalDescent', 'AverageSpeed', 'MaximumSpeed', \
            'AveragePower', 'Difficulty', 'MinimumTemperature', \
            'MaximumTemperature', 'MinimumAltitude', 'MaximumAltitude', \
            'MovingTime', 'ElapsedTime', 'Time']
  for var in floats:
      data[var] = data[var].astype('float')

  to_delete = ['Cardio', 'Yoga', 'Athletic walking', 'Athletic Walking', 'Alpinism', 'Swimming', 'Hike', 'Walk']
  data.drop(data[data['ActivityType'].isin(to_delete)].index, inplace = True)
  return data



def get_performance(mu,sigma,min,max):
  '''Function that, given a mu and sigma, returns a number between min and max that follows a gausian distribution'''

  perfo = random.gauss(mu, sigma)
  i = 0
  while perfo < min or max < perfo:
    i += 1
    np.random.seed(i)
    perfo = random.gauss(mu, sigma)
  return float(perfo)

def performance(df, data, separator):
  '''Function that, given a separator (indicates which parameters are used to
  infer the performance for each activity type) and dataset, returns the dataset
  with the performance infered for each activity'''

  min = 1
  max = 10
  avg = pd.DataFrame(data=df, columns=['Performance', 'Date'], index=df.index)
  i = 0
  for row in separator:
    if row not in df['ActivityType'].unique():
        pass
    else:
      aux = df[df['ActivityType'] == row]
      aux2 = data[data['ActivityType'] == row]
      for var in separator[row]:
          aux[var] =((aux[var] - np.min(aux2[var]))/(np.max(aux2[var]) - np.min(aux2[var])))*(max-min)+min
      for index in list(aux.index):
        prev = 0
        for i in range (20):
          if row == 'Mountain biking':
            mu = (aux.loc[index, 'Distance'] - aux.loc[index, 'AverageHeartRate'] + aux.loc[index, 'AverageSpeed'] + aux.loc[index, 'TotalAscent'] + aux.loc[index, 'Calories'] + aux.loc[index, 'MovingTime'] + aux.loc[index, 'MaximumAltitude'] - (aux.loc[index, 'ElapsedTime'] - aux.loc[index, 'MovingTime']))/8
          elif row == 'Road biking':
            mu = (aux.loc[index, 'Distance'] + aux.loc[index, 'AveragePower'] - aux.loc[index, 'AverageHeartRate'] + aux.loc[index, 'AverageSpeed'] + aux.loc[index, 'TotalAscent'] + aux.loc[index, 'Calories'] + aux.loc[index, 'MovingTime'] + aux.loc[index, 'MaximumAltitude'] - (aux.loc[index, 'ElapsedTime'] - aux.loc[index, 'MovingTime']))/9
          elif row == 'Spinning':
            mu = (aux.loc[index, 'ElapsedTime'] - aux.loc[index, 'AverageHeartRate'] + aux.loc[index, 'AveragePower'] + aux.loc[index, 'Calories'])/4
          elif row == 'Running':
            mu = (aux.loc[index, 'Distance'] - aux.loc[index, 'AverageHeartRate'] + aux.loc[index, 'AverageSpeed'] + aux.loc[index, 'TotalAscent'] + aux.loc[index, 'Calories'] + aux.loc[index, 'MovingTime'] + aux.loc[index, 'MaximumAltitude'] - (aux.loc[index, 'ElapsedTime'] - aux.loc[index, 'MovingTime']))/8
          else:
            mu = (aux.loc[index, 'Calories'] - (aux.loc[index, 'ElapsedTime'] - aux.loc[index, 'MovingTime']) + aux.loc[index, 'MovingTime'] + aux.loc[index, 'AverageHeartRate'] + aux.loc[index, 'MaximumHeartRate'])/5
          sigma = (max-min)/4
          next = get_performance(mu, sigma, min, max)
          if i == 0:
            avg.loc[index, 'Performance'] = next
          else:
            avg.loc[index, 'Performance'] = (prev + next)/2
          prev = avg.loc[index, 'Performance']
        df.loc[index, 'Performance'] = avg.loc[index, 'Performance']
  return df.reset_index(drop = True)

def group_by_weeks(data):
  res = data.copy()
  res = res.fillna(0)
  res['Date'] = pd.to_datetime(res['Date'])
  res['day-of-week'] = res['Date'].dt.day_name()
  res['Week'] = pd.to_datetime(pd.to_datetime(res['Date']).dt.date)
  days = ['Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday' ]
  for i in range(6):
    res.loc[res["day-of-week"] == days[i], "Week"] = res.Date - pd.to_timedelta(i+1, unit='d')
    i = i+1
  count = res.groupby('Week').count()
  counts = res.copy()
  res['Week'] = res['Week'].dt.date
  res = res.groupby('Week').mean()

  #Add extra features to data
  res['Count'] = count['ActivityType']
  res['Week'] = pd.to_datetime(res.index)
  res['Month'] = res.Week.dt.month
  res['Year'] = res.Week.dt.year
  #Add number of aactivitytypes per week:
  counts = pd.DataFrame(counts.groupby(['Week', 'ActivityType']).size()).reset_index().rename(columns = { 0:'Counts'})
  activities = ['Running', 'Road biking', 'Mountain biking', 'Weight training', 'Spinning']
  for act in activities:
    counts[act] = counts[counts['ActivityType'] ==  act].Counts
  counts = counts.groupby('Week').sum()
  for act in activities:
    res[act] = counts[act]

  return res

def perf_label(df, data, weekly):
  separator = {'Mountain biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'Calories', 'MaximumAltitude'],
              'Road biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'AveragePower', 'Calories', 'MaximumAltitude'],
              'Spinning': ['ElapsedTime', 'AverageHeartRate', 'AveragePower', 'Calories'],
              'Running': ['Distance', 'AverageHeartRate', 'AverageSpeed', 'ElapsedTime', 'MovingTime', 'TotalAscent', 'Calories', 'MaximumAltitude'],
              'Weight training': ['Calories', 'ElapsedTime', 'AverageHeartRate', 'MaximumHeartRate', 'MovingTime']}

  df = performance(df, data, separator)
  new_weeks = group_by_weeks(df)

  return df, new_weeks

def recomendation(new_weeks, weekly, data, result):
  '''Given a dataset, and a weekly training routine,
  saves the recomensation messages in the weekly dataset.'''

  # Save the activities corresponding to the positive, negative and maintenance weeks
  positive = data[data['Week'].isin(np.array(weekly[weekly['Label'] == 'Positive'].Week))]
  negative = data[data['Week'].isin(np.array(weekly[weekly['Label'] == 'Negative'].Week))]
  maintenance = data[data['Week'].isin(np.array(weekly[weekly['Label'] == 'Maintenance'].Week))]

  # Save the Pos/Neg/Maint weeks
  pos_week = weekly[weekly['Label'] == 'Positive']
  neg_week = weekly[weekly['Label'] == 'Negative']
  main_week = weekly[weekly['Label'] == 'Maintenance']

  # Save conditions
  c1 = ((new_weeks['Time'] > pos_week['Time'].mean() + 2*pos_week['Time'].std()) | \
      (new_weeks['ElapsedTime'] > pos_week['ElapsedTime'].mean() + 2*pos_week['ElapsedTime'].std()) | \
      (new_weeks['MovingTime'] > pos_week['MovingTime'].mean() + 2*pos_week['MovingTime'].std()))

  c2 = new_weeks['Count'] > pos_week['Count'].mean() + 2*pos_week['Count'].std() # if count >> mean(count of positive)

  c3 = new_weeks['AverageHeartRate'] > pos_week['AverageHeartRate'].mean() + pos_week['AverageHeartRate'].std()

  c4 = ((new_weeks['Time'] < pos_week['Time'].mean() - pos_week['Time'].std()) | \
      (new_weeks['ElapsedTime'] < pos_week['ElapsedTime'].mean() - 2*pos_week['ElapsedTime'].std()) | \
      (new_weeks['MovingTime'] < pos_week['MovingTime'].mean() - 2*pos_week['MovingTime'].std()))

  c5 = new_weeks['Count'] < pos_week['Count'].mean() - pos_week['Count'].std() # if count >> mean(count of positive)

  c6 = new_weeks['AverageHeartRate'] < pos_week['AverageHeartRate'].mean() - pos_week['AverageHeartRate'].std()

  c7 = ((new_weeks['Road biking']/new_weeks['Count'] > 0.7) | \
      (new_weeks['Mountain biking']/new_weeks['Count']  > 0.7) | \
      (new_weeks['Spinning']/new_weeks['Count']  > 0.7) | \
        (new_weeks['Weight training']/new_weeks['Count']  > 0.7) | \
        (new_weeks['Running']/new_weeks['Count']  > 0.7))

  c8 = new_weeks['TotalAscent'] < pos_week['TotalAscent'].mean() - pos_week['TotalAscent'].std()

  c9 = (new_weeks['Label'] == 'Negative')

  # Save message for each specific week
  conditions = {'m1':c1, 'm2':c2, 'm3':c3, 'm4':c4, 'm5':c5, 'm6':c6, 'm7':c7, 'm8':c8, 'm9':c9}
  for m in conditions:
      new_weeks[m] = np.where(conditions[m], 1, 0)

  return new_weeks

def compare(weekly, new_weeks):
    max = np.max(weekly['Performance'])
    now = new_weeks['Performance']
    new_weeks['Percentage'] = (now*100)/max
    return new_weeks


def gauge(perc):
    '''Function that given a value shows a gauge chart.'''

    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = perc,
    number={'font': {'size': 40}},
    gauge = {
    'axis': {'range': [0, max(100, perc)], 'tickwidth': 1},
    'bar': {'color': "grey", 'thickness': 0.05},
    'borderwidth': 0,
    'steps': [
    {'range': [0, 100/3], 'color': '#F01B06'},
    {'range': [100/3, (200)/3], 'color': '#FD9C02'},
    {'range': [200/3, max(100, perc)], 'color': '#006400'}
    ]}))
    fig.update_layout(
    #font={'color': "grey", 'family': "Calibri", 'size': 15},
    title={'text': "CURRENT PHYSISCAL CONDITION",
    'font': {'size': 30, 'color': "grey"},
    'y': 0.95,
    'x': 0.5,
    'xanchor': 'center'},
    margin={ 'l':120, 'r':100 })
    st.write(fig)


def show(new_weeks):
    ''' Function that, given new weeks of trainings, displays the results.'''

    with st.columns([1, 3, 1])[1]:
        # One tab for each new week.
        if len(new_weeks.index) == 0:
            err = "The results of the selected week were not correctly stored because they were ambiguous and confusing the predictive model."
            st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:20px; border-radius:2%;">{err}</p>', unsafe_allow_html=True)
        else:
            tabs = st.tabs([str(week) for week in new_weeks.index])
            for idx,tab in enumerate(tabs):
                with tab:
                    wk = new_weeks.iloc[idx, :]
                    lab = wk['Label']
                    perc = wk['Percentage']

                    # Show the gauge chart:
                    #st.markdown(f'<p style="text-align: center; color:grey; font-size:30px;">CURRENT PHYSISCAL CONDITION</p>', unsafe_allow_html=True)
                    gau = gauge(perc)

                    # Show the resulting label:

                    text =  ("THIS WEEK'S TRAINING ROUTINE HAS BEEN " + lab.upper() + "  \n ")
                    if lab == 'Positive':
                        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#D8F1B5; color:#006400; font-size:20px; border-radius:2%;">{text}</p>', unsafe_allow_html=True)
                    elif lab == 'Negative':
                        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:20px; border-radius:2%;">{text}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5E4C9; color:#FD9C02; font-size:20px; border-radius:2%;">{text}</p>', unsafe_allow_html=True)

                    # Show the recommendation messages:
                    m1 = "Rest. Too much minutes"
                    m2 = "Rest. Too much activities"
                    m3 = "Rest. Too much anaerobic"
                    m4 = "Try to do longer activities"
                    m5 = "Try to train more days"
                    m6 = "Try to do more anaerobic activities"
                    m7 = "Try different activities"
                    m8 = "Would be a good idea to train by accumulating more positive elevation gain"
                    m9 = "Unlucky your performance will probably descrease. Try to change your training routine"
                    messages = {'m1':m1, 'm2':m2, 'm3':m3, 'm4':m4, 'm5':m5, 'm6':m6, 'm7':m7, 'm8':m8, 'm9':m9}

                    text = ""
                    i = 0
                    for m in messages:
                        if wk[m]:
                            text = text + "  \n" + str(i) + " - "  + messages[m]
                            i = i+1
                    st.write("  \n")
                    st.write("  \n")
                    st.write("  \n")
                    st.markdown(f'<p style="color:#303030; font-size:25px;">RECOMMENDATION MESSAGES:</p>', unsafe_allow_html=True)
                    st.code(text, language="markdown")

def predict(new_weeks, weekly):
    '''Function that, given a dataset with new_weeks of training, returns the label for each one. '''
    try:
        model = joblib.load('Model/Model.sav')
        with open('Model/key.txt') as f:
            features = [line.strip() for line in f]
        scaler = StandardScaler().fit(weekly[features].fillna(0))
        X = scaler.transform(new_weeks[features].fillna(0))
        result = model.predict(X)
        new_weeks['Label'] = result
    except:
        no_req = "THERE IS NOT ANY SAVED MODEL. YOU CAN SOLVE THIS BY SAVING THE MODEL IN THE PREVIOUS SECTION."
        st.markdown(f'<p style="text-align: center; padding: 20px; background-color:#F5CDC9; color:#F01B06; font-size:15px; border-radius:2%;">{no_req}</p>', unsafe_allow_html=True)

    return new_weeks, result

def search_prev(dy, weekly, data):
    '''Function that returns the dataset with the activities of the week of
    the selected date and the corresponding week.'''

    w = dy - pd.to_timedelta(dy.weekday() , unit='d')
    prev_weeks = weekly[weekly.index == str(w)]
    prev_weeks = compare(weekly, prev_weeks)
    prev_weeks = recomendation(prev_weeks, weekly, data, prev_weeks['Label'])
    return prev_weeks, w


def save(data, weekly, new_weeks, df):
    show(new_weeks)
    data = data.append(df)
    weekly = weekly.append(new_weeks)
    weekly = weekly.fillna(0)
    data = data.fillna(0)
    data.to_csv('Data/data.csv')
    weekly.to_csv('Data/weekly.csv')
    #df = pd.read_csv('Data/data.csv') #
    st.write("Data saved :heavy_check_mark:")
    for key in st.session_state.keys():
        del st.session_state[key]

def footnote_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
