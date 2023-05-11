from config import*

def read(data):
  '''Function that reads the dataset exported from GARMIN APP'''

  colnames=["ActivityType", "Date", "Favourite", "Title", "Distance",
            "Calories", "Time", "AverageHeartRate", "MaximumHeartRate",
            "AerobicTrainingEffect", "AverageRunningCadence",
            "MaximumRunningCadence", "AverageSpeed", "MaximumSpeed",
            "TotalAscent", "TotalDescent", "AverageStrideLength",
            "AverageVerticalRatio", "AverageVerticalOscillation",
            "AverageGroundContactTime", "AveragePedalingCadence",
            "MaximumPedalingCadence", "NormalizedPower_NP_", "LeftRightBalance",
            "TrainingStressScore", "MaximumAveragePower_20Minutes_",
            "AveragePower", "MaximumPower", "Difficulty", "Fluency",
            "TotalStrokes", "AverageSwolf", "AverageStrokeSpeed",
            "TotalRepetition", "TotalSeries", "DiveTime", "MinimumTemperature",
            "SurfaceInterval", "Decompression", "BestLapTime", "NumberOfTurns",
            "MaximumTemperature", "AverageRateOfRespiration",
            "MinimumRateOfRespiration", "MaximumBreathingRate", "ChangeInStress",
            "BeginningOfTheStressPeriod", "EndOfStressPeriod", "AverageStress",
            "MovingTime", "ElapsedTime", "MinimumAltitude", "MaximumAltitude"]

  data = pd.read_csv(data, sep = ';')
  return data


def clean(data):
  '''Function that given the raw dataset from GARMIN APP, cleans it'''

  data = data.replace('"--"','"NaN"')

  #Make all values equal in order to clean them a posteriori (add "")
  not_str = ['Calories', 'Distance', 'TotalAscent', \
            'TotalDescent', 'MinimumAltitude', 'MaximumAltitude', 'Date']
  for var in not_str:
    data.loc[data[var].str[0] != '"', var] = ('"'+data[var]+'"')

  #To str without ""
  data["Title"] = (data['Title'].astype('str')).str[1:-1]
  data['Date'] = pd.to_datetime((data[var].astype('str')).str[1:-1])

  #To Floats
  floats = ['Distance', 'Calories', 'AverageHeartRate', 'MaximumHeartRate', \
            'AerobicTrainingEffect', 'TotalAscent', 'TotalDescent', \
            'AveragePower', 'Difficulty', 'MinimumTemperature', \
            'MaximumTemperature', 'MinimumAltitude', 'MaximumAltitude']
  for var in floats:
    data[var] = ((data[var].astype('str')).str[1:-1]).astype('float')

  #To minutes (not datetime)
  to_minutes = ['MovingTime', 'ElapsedTime', 'Time']
  for var in to_minutes:
    data[var] = ((data[var].astype('str')).str[1:3]).astype('int')*60 + \
    ((data[var].astype('str')).str[4:6]).astype('int')

  #from pace (min/km) to speed (km/h)
  to_speed = ['MaximumSpeed', 'AverageSpeed']
  for pace in to_speed:
    cond1 = ((data[pace].str.contains(':')) & (data[pace].str.len() >= 7 ))
    cond2 = ((data[pace].str.contains(':')) & (data[pace].str.len() < 7 ))

    speed1 = '"'+(3600/((data.loc[cond1, pace].str[1:3]).astype(float)*60 + \
            (data.loc[cond1, pace].str[4:6]).astype(float))).astype('str')+'"'
    speed2 = '"'+(3600/((data.loc[cond2, pace].str[1:2]).astype(float)*60 + \
            (data.loc[cond2, pace].str[3:5]).astype(float))).astype('str')+'"'

    data.loc[cond1, pace] = speed1
    data.loc[cond2, pace] = speed2

    #from str to float
    data[pace] = ((data[pace].astype('str')).str[1:-1]).astype('float')
  #From spanish to English:
  data['ActivityType'] = data['ActivityType'].replace([\
                        'Ciclismo en ruta','Ciclismo de montaña', 'Ciclismo',\
                        'Ciclismo en sala', 'Entreno de fuerza', 'Carrera', \
                        'Marcha', 'Montañismo', 'Natación en aguas abiertas',\
                        'Caminar', 'Paseo'],\
                        ['Road biking', 'Mountain biking', 'Road biking',\
                          'Spinning','Weight training', 'Running',\
                          'Athletic Walking', 'Alpinism', 'Swimming','Hike', 'Walk'])

  #Delete usless columns:
  data = data.drop(['AverageRunningCadence', 'MaximumRunningCadence',
                    'AverageStrideLength', 'AverageVerticalRatio',
                    'AverageVerticalOscillation', 'AverageGroundContactTime',
                    'AveragePedalingCadence', 'MaximumPedalingCadence',
                    'NormalizedPower_NP_', 'LeftRightBalance',
                    'TrainingStressScore', 'MaximumAveragePower_20Minutes_',
                    'MaximumPower', 'Fluency', 'TotalStrokes', 'AverageSwolf',
                    'AverageStrokeSpeed', 'TotalRepetition', 'TotalSeries',
                    'DiveTime', 'SurfaceInterval', 'Decompression', 'BestLapTime',
                    'NumberOfTurns', 'AverageRateOfRespiration',
                    'MinimumRateOfRespiration', 'MaximumBreathingRate',
                    'ChangeInStress', 'BeginningOfTheStressPeriod',
                    'EndOfStressPeriod', 'AverageStress', 'Favourite', 'AerobicTrainingEffect'], axis=1)
  to_delete = ['Cardio', 'Yoga', 'Athletic walking', 'Athletic Walking', 'Alpinism', 'Swimming', 'Hike', 'Walk']
  data.drop(data[data['ActivityType'].isin(to_delete)].index, inplace = True)
  return data


def remove_outliers_quartiles(data):
  '''Function that removes outliers using quartiles method'''

  separator = {'Mountain biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'Calories', 'MaximumAltitude'],
              'Road biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'AveragePower', 'Calories', 'MaximumAltitude'],
              'Spinning': ['ElapsedTime', 'AverageHeartRate', 'AveragePower', 'Calories'],
              'Running': ['Distance', 'AverageHeartRate', 'AverageSpeed', 'ElapsedTime', 'MovingTime', 'TotalAscent', 'Calories', 'MaximumAltitude'],
              'Weight training': ['Calories', 'ElapsedTime', 'AverageHeartRate', 'MaximumHeartRate', 'MovingTime']}

  for row in separator:
    aux = data[data['ActivityType'] == row]
    for var in separator[row]:
      to_remove = aux[var].between(aux[var].quantile(.02), aux[var].quantile(.98))
      index_names = aux[~to_remove].index
      for i in range(len(index_names)):
        if index_names[i] in data.index:
          data = data.drop(index_names[i])
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


def performance(data, separator):
  '''Function that, given a separator (indicates which parameters are used to
  infer the performance for each activity type) and dataset, returns the dataset
  with the performance infered for each activity'''

  min = 1
  max = 10
  avg = pd.DataFrame(data=data, columns=['Performance', 'Date'], index=data.index)

  for row in separator:
      aux = data[data['ActivityType'] == row]
      for var in separator[row]:
        aux[var] =((aux[var] - np.min(aux[var]))/(np.max(aux[var]) - np.min(aux[var])))*(max-min)+min
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
        data.loc[index, 'Performance'] = avg.loc[index, 'Performance']
  return data.reset_index(drop = True)



#GROUP BY WEEKS & LABEL THE GROUPED DATA
def group_by_weeks(data):
    '''Function that given a datetime dataset, returns the same dataset grouped by weeks.
    It also adds extra features to the original dataset.'''
    res = data.copy()
    data = data.fillna(0)
    data['day-of-week'] = data['Date'].dt.day_name()
    data['Week'] = pd.to_datetime(data['Date'])
    days = ['Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday' ]
    for i in range(6):
        data.loc[data["day-of-week"] == days[i], "Week"] = data.Date - pd.to_timedelta(i+1, unit='d')
        i = i+1
        data['Week'] = pd.to_datetime(pd.to_datetime(data['Week']).dt.date)

  #Create the weekly grouped dataset:
    weekly = data.copy()
    count = weekly.groupby('Week').count()
    counts = weekly.copy()
    weekly = weekly.drop(['ActivityType', 'Date', 'Title', 'day-of-week'], axis = 1).groupby('Week').mean()
    weekly = weekly.groupby('Week').mean()

  #Add extra features to weekly grouped data
    weekly['Count'] = count['ActivityType']
    weekly['Week'] = weekly.index
    weekly['Month'] = weekly.Week.dt.month
    weekly['Year'] = weekly.Week.dt.year

  #Add number of aactivitytypes per week:
    counts = pd.DataFrame(counts.groupby(['Week', 'ActivityType']).size()).reset_index().rename(columns = { 0:'Counts'})
    activities = data['ActivityType'].unique()
    for act in activities:
        counts[act] = counts[counts['ActivityType'] ==  act].Counts
    counts = counts.groupby('Week').sum()
    for act in activities:
        weekly[act] = counts[act]

    return weekly, data

def label(weekly):
  weekly['Label'] = 'to_fill'
  for i in range(len(weekly)-1):
    next = weekly['Performance'].iloc[i+1]
    if next-0.25 <= weekly['Performance'].iloc[i] <= next+0.25:
      weekly['Label'].iloc[i] = 'Maintenance'
    elif next > weekly['Performance'].iloc[i]:
      weekly['Label'].iloc[i] = 'Positive'
    else:
      weekly['Label'].iloc[i] = 'Negative'
  #Remove last row since we cannot compare it to nothing
  weekly = weekly.iloc[: len(weekly)-1, :]
  return weekly

@st.cache_data(show_spinner=False)
def perf_label(data, weeklyavg):
  separator = {'Mountain biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'Calories', 'MaximumAltitude'],
              'Road biking': ['Distance', 'ElapsedTime', 'AverageHeartRate', 'AverageSpeed', 'TotalAscent', 'MovingTime', 'AveragePower', 'Calories', 'MaximumAltitude'],
              'Spinning': ['ElapsedTime', 'AverageHeartRate', 'AveragePower', 'Calories'],
              'Running': ['Distance', 'AverageHeartRate', 'AverageSpeed', 'ElapsedTime', 'MovingTime', 'TotalAscent', 'Calories', 'MaximumAltitude'],
              'Weight training': ['Calories', 'ElapsedTime', 'AverageHeartRate', 'MaximumHeartRate', 'MovingTime']}

  check1 = pd.DataFrame(data=None, columns=['L1', 'L2', 'L3', 'L4', 'L5'], index=weeklyavg.index)
  check2 = pd.DataFrame(data=None, columns=['P1', 'P2', 'P3', 'P4', 'P5'], index=data.index)

  for i in range(5):
    #performance is the average of 20 times
    data = performance(data, separator)

    #each time the performance is different in the groupby
    weeklyavg, data = group_by_weeks(data)
    weekly = label(weeklyavg)
    st.write(check1)
    st.write(check2)
    check1.iloc[:,i] = weekly['Label']

    check2.iloc[:,i] = data['Performance']

  #update the performance of original data with avg of each execution
  check2['Performance'] = check2.mean(axis = 1)
  data['Performance'] = check2['Performance']
  #also update it in the goruped data
  weeklyavg, data = group_by_weeks(data)

  #Count the number of each result for each week.
  check1['Pos'] = check1[check1 == 'Positive'].count(axis=1)
  check1['Neg'] = check1[check1 == 'Negative'].count(axis=1)
  check1['Maint'] = check1[check1 == 'Maintenance'].count(axis=1)

  #Label according to the discrepancies
  check1.loc[(check1['Pos']>1) & (check1['Neg']>1),'Label'] = 'Ambiguous'
  check1.loc[(check1['Pos']>check1['Maint']) & (check1['Neg']<=1),'Label'] = 'Positive'
  check1.loc[(check1['Neg']>check1['Maint']) & (check1['Pos']<=1),'Label'] = 'Negative'
  check1.loc[(check1['Maint']>=check1['Neg']) & (check1['Maint']>=check1['Pos']),'Label'] = 'Maintenance'

  #Add the final label to the weekly dataset
  weeklyavg['Label'] = check1['Label']

  #Delete rows that are ambiguous
  ambiguous = weeklyavg[weeklyavg.Label == 'Ambiguous']
  weeklyavg = weeklyavg.drop(weeklyavg[weeklyavg.Label == 'Ambiguous'].index)


  #return the resulting datasets
  return data, weeklyavg


def check(weekly):
    class_counts = weekly['Label'].value_counts()
    # Compute the ratio of the smallest class to the largest class
    class_ratios = [class_counts.min() / class_counts.max(), class_counts.max() / class_counts.min()]

    # Define the range of acceptable ratios
    acceptable_range = [0.5, 1.3]

    # Check if the ratio falls within the acceptable range
    if (class_ratios[0] >= acceptable_range[0] and class_ratios[1] <= acceptable_range[1]):
        return True
    return False
        #df = generate(data, result_dict)
        #data = data.append(df, ignore_index=True)



def get_gen(weekly):
    my_dict = dict(weekly['Label'].value_counts())
    min_keys = sorted(my_dict, key=my_dict.get)[:2]
    max_value = np.max(list(my_dict.values()))
    # Dictonaru that has each label and how many registers to generate for each
    result_dict = {key: max_value - value for key, value in my_dict.items() if key in min_keys}
    return result_dict




def generate_row(data, lab, act, date):
  dif = ['AverageHeartRate', 'MaximumHeartRate', 'MinimumTemperature', 'MaximumTemperature', 'MinimumAltitude', 'MaximumAltitude']
  if lab == 'Positive':
    #positive weeks
    new_row =  {'ActivityType':act, 'Date': date, 'Title': act + " activity"}
    for col in data.columns:
     if col in ['ActivityType','Date', 'Title']:
        pass
     elif col not in(dif):
       new_row[col] = random.uniform(np.mean(data.loc[data['ActivityType'] == act, col]), np.max(data.loc[data['ActivityType'] == act, col]))
     else:
       new_row[col] = random.uniform(np.min(data.loc[data['ActivityType'] == act, col]), np.mean(data.loc[data['ActivityType'] == act, col]))

  elif lab == 'Negative':
    #negative weeks
    new_row =  {'ActivityType':act, 'Date': date, 'Title': act + " activity"}
    for col in data.columns:
     if col in ['ActivityType','Date', 'Title']:
        pass
     elif col not in(dif):
       new_row[col] = random.uniform(np.min(data.loc[data['ActivityType'] == act, col]), np.mean(data.loc[data['ActivityType'] == act, col]))
     else:
       new_row[col] = random.uniform(np.mean(data.loc[data['ActivityType'] == act, col]), np.max(data.loc[data['ActivityType'] == act, col]))

  else:
      # Maintenance weeks:
    new_row =  {'ActivityType':act, 'Date': date, 'Title': act + " activity"}
    return
    for col in data.columns:
      if col in ['ActivityType','Date', 'Title']:
        pass
      else:
        new_row[col] = random.uniform(np.mean(data.loc[data['ActivityType'] == act, col]) - np.std(data.loc[data['ActivityType'] == act, col]),
                                      np.mean(data.loc[data['ActivityType'] == act, col]) + np.std(data.loc[data['ActivityType'] == act, col]))
  return new_row


def generate(data, result_dict):
  # Labels that need to be generated
  g1, g2 = list(result_dict.keys())

  # Dict with labels that need tot be generated and the already generated
  done = dict.fromkeys(list(result_dict.keys()), 0)

  # Defining the dates.
  data['Date'] = pd.to_datetime(data['Date'])
  #Make sure each week is generated within the same week. Make sure it is a Sat.
  date = np.min(data.Date)-dt.timedelta(days=7)
  date = date + dt.timedelta(days=5) - dt.timedelta(days=date.day_of_week)


  # Initualize the new dataframe
  df = pd.DataFrame(columns = data.columns)

  # Start generating
  while done[g1] < result_dict[g1] or done[g2] < result_dict[g2]:
    # Generate weeks of g1 or g2 randomly. lab == 1 --> generate for g1
    lab = random.randint(1,2)
    act = random.choice(data['ActivityType'].unique())
    # If we have already generated all needed weeks for g1, or lab == 2, then we generate for g2
    if done[g1] == result_dict[g1] or lab == 2:
      done[g2] = done[g2] + 1
      #generate weeks for g2
      for i in range(3):
        act = random.choice(data['ActivityType'].unique())
        new_row = generate_row(data, g2, act, date-dt.timedelta(days=2*i))
        df = df.append(new_row, ignore_index=True)
    else:
      done[g1] = done[g1] + 1
  #    #generate weeks for g1
      for i in range(3):
        act = random.choice(data['ActivityType'].unique())
        new_row = generate_row(data, g1, act, date-dt.timedelta(days=2*i))
        df = df.append(new_row, ignore_index=True)
    date = date - dt.timedelta(days=7)
  return df

@st.cache_data(show_spinner="Balancing the dataset...")
def balance(weekly, data, _feat):
    gen = get_gen(weekly)
    df = generate(data[_feat], gen)
    # Append generated weeks to the original dataset
    data = data.append(df, ignore_index=True)

    # Check if now is balanced
    weekly, data = group_by_weeks(data)
    data, weekly = perf_label(data, weekly)
    bal = check(weekly)
    return bal, data, weekly




def reduced_model(model, features, weeklyavg, importance):
  aux = {}
  j = 0
  for coef in importance:
    aux[features[j]] = coef
    j = j+1
  res = dict(sorted(aux.items(), key = itemgetter(1), reverse = True)[:10])
  keysList = list(res.keys())
  X = weeklyavg[keysList]
  y = weeklyavg["Label"]
  X = StandardScaler().fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
  return model.fit(X_train, y_train), X_train, X_test, y_train, y_test, keysList

#st.cache_resource(show_spinner="Finding the best model")
def train(weekly):
  '''Given a weekly grouped dataset, the function trains 6 classificators and
  returns a list with the accuracy metrics for each one.'''

  X = weekly.drop(["Label", 'Week'],1)   # Feature Matrix
  y = weekly["Label"]          # Target Variable

  # Create training and test split
  #X = X.fillna(0)
  features = X.columns
  X = StandardScaler().fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

  # Model pipeline
  model_pipeline = []
  model_pipeline.append(LogisticRegression(solver='liblinear'))
  model_pipeline.append(SVC())
  model_pipeline.append(KNeighborsClassifier())
  model_pipeline.append(DecisionTreeClassifier())
  model_pipeline.append(RandomForestClassifier())
  model_pipeline.append(GaussianNB())

  # Model Evaluation
  model_list = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
  acc_list = []
  cm_list = []
  acc_list2 = []
  cm_list2 = []
  i = 0
  max = [0, 0]
  X_train2_ret = []
  X_test2_ret = []
  y_train2_ret = []
  y_test2_ret = []
  keylist_ret = []
  y_pred2_ret = []
  for model in model_pipeline:
    model.fit(X_train, y_train)
    if i == 0:
      importance = model.coef_[0]

    elif i == 3 or i == 4:
      importance = model.feature_importances_

    else:
      perm_importance = permutation_importance(model, X_test, y_test)
      importance = perm_importance.importances_mean

    i = i+1

    #Preduction with all the features
    y_pred = model.predict(X_test)

    #Prediction with the 10 most important features
    model2, X_train2, X_test2, y_train2, y_test2, keylist = reduced_model(model, features, weekly, importance)
    y_pred2 = model2.predict(X_test2)

    #Append the result to the lists
    acc_list.append(accuracy_score(y_test, y_pred))
    cm_list.append(confusion_matrix(y_test, y_pred))
    acc_list2.append(accuracy_score(y_test2, y_pred2))
    cm_list2.append(confusion_matrix(y_test2, y_pred2))

    #Save the best model:
    if accuracy_score(y_test2, y_pred2) > max[0] and recall_score(y_test2, y_pred2, average='macro') > max[1]:
      max[0] = accuracy_score(y_test2, y_pred2)
      max[1] = recall_score(y_test2, y_pred2, average='macro')
      mod = model2
      X_train2_ret, X_test2_ret, y_train2_ret, y_test2_ret, keylist_ret, y_pred2_ret =  X_train2, X_test2, y_train2, y_test2, keylist, y_pred2


  return max, pd.DataFrame({'Model': model_list, 'Accuracy1': acc_list, 'Accuracy2': acc_list2}), mod, X_train2_ret, X_test2_ret, y_train2_ret, y_test2_ret, keylist_ret, y_pred2_ret

#st.cache_resource(show_spinner="Finding the best model")
def find_model(weekly, acc, rec, ti):
    ret = [0, 0]
    timeout = Time.time() + 60*ti
    while acc > ret[0] or rec > ret[1]:
        if Time.time() > timeout:
            return ret, df, model, X_train, X_test, y_train, y_test, keylist, y_pred, False
        ret, df, model, X_train, X_test, y_train, y_test, keylist, y_pred = train(weekly)
    return ret, df, model, X_train, X_test, y_train, y_test, keylist, y_pred, True

def save(model, X_train, X_test, y_train, y_test, keylist, y_pred, weekly, data):
    pickle.dump(model, open('Model/Model.sav', 'wb'))
    np.save('Model/X_train', X_train)
    np.save('Model/X_test', X_test)
    np.save('Model/y_train', y_train)
    np.save('Model/y_test', y_test)
    np.save('Model/y_pred', y_pred)
    np.savetxt("Model/key.txt", keylist, delimiter=",", newline = "\n", fmt="%s")
    #weekly.to_csv("Data/weekly.csv")
    #data.to_csv("Data/data.csv")
    for key in st.session_state.keys():
        del st.session_state[key]


def metrics(weekly, model, X_train, X_test, y_train, y_test, keylist, y_pred):
    # METRICS
    mod1 = "MODEL FOUND: " + (str(model)[:-2])
    mod2 = (str(model)[:-2])
    st.markdown(f'<p style="text-align: center; padding: 10px; background-color:#D8F1B5; color:#006400; font-size:20px; border-radius:2%;">{mod1}</p>', unsafe_allow_html=True)
    results = pd.DataFrame(y_pred, y_test).reset_index()
    res1 = pd.DataFrame(classification_report(results.iloc[:,0], results.iloc[:,1], output_dict=True)).transpose()
    res2 = pd.DataFrame(res1[(res1.index == 'weighted avg') | (res1.index == 'macro avg')])
    res2['Accuracy'] = res1.iloc[3, 1]


    # CONFUSION MATRIX
    labels = ['Positive', 'Negative', 'Maintenance']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_df = pd.DataFrame(disp.confusion_matrix, index=disp.display_labels, columns=disp.display_labels)

    # Convert the DataFrame to long format for Altair
    cm_df_long = cm_df.stack().reset_index()
    cm_df_long.columns = ['true_label', 'predicted_label', 'count']

    base = alt.Chart(cm_df_long).encode(
    x=alt.X('true_label:N', sort=labels, title = "TRUE LABEL"),
    y=alt.Y('predicted_label:N', sort=labels, title = "PREDICTED LABEL")
    ).properties(
    width = 500,
    height = 400
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('count:Q')
    )

    text = base.mark_text(baseline='middle').encode(
        alt.Text('count:Q', format=".0f")
    )
    return res1, res2, (heatmap + text)


def create_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def footnote_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
