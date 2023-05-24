from config import*

def read(username, s3):
    k1 = f"Data/{username}/Data.csv"
    k2 = f"Data/{username}/Weekly.csv"

    response1 = s3.get_object(Bucket='fitnessapdata', Key=k1)
    csv_contents1 = response1['Body'].read().decode('utf-8')
    response2 = s3.get_object(Bucket='fitnessapdata', Key=k2)
    csv_contents2 = response2['Body'].read().decode('utf-8')

    # Convert the CSV data into a Pandas DataFrame
    data = pd.read_csv(StringIO(csv_contents1))
    weekly = pd.read_csv(StringIO(csv_contents2), index_col = 0).rename(columns={'Week.1': 'Week'})
    return data, weekly

def clean3(data):
  '''Function that given the raw dataset from GARMIN APP, cleans it'''

  # From str to datetime
  data['Date'] = pd.to_datetime(data['Date'])

  # New usefull features for the visualitzations:
  data['Year'] = pd.to_datetime(data['Date']).dt.year
  data['Month'] = pd.to_datetime(data['Date']).dt.month
  data['MonthName'] = data['Date'].dt.month_name()
  return data

def perf_evol(data, year):
    chart = alt.Chart(data[data['Year'] == year]).mark_line(
    point=alt.OverlayMarkDef(filled=False, fill="white")
    ).encode(
    alt.X('month(Date):T', title = None),
    alt.Y('mean(Performance):Q')
    ).properties(
        width=600,
        height=300
    )
    return chart

def count_activities(data):
    # create a bar chart using Altair
    chart = alt.Chart(data).mark_bar().encode(
        y=alt.Y('count():Q', title = 'counts'),
        x=alt.X('ActivityType:N', sort=alt.EncodingSortField(field="count():Q", op = "count", order = "descending"), title = None)
    ).properties(
        width=215,
        height=525
    )
    return chart

def count_act_month(data):
    counts = data.groupby(['MonthName', 'ActivityType']).size().reset_index(name='counts')

    chart = alt.Chart(counts).mark_bar().encode(
    column=alt.Column(
    'MonthName',
    header=alt.Header(orient='bottom',  labelColor='grey', labelFontSize = 12, labelAngle=0, labelPadding = 28), title = None,
    sort=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ),
    x=alt.X('ActivityType', axis=alt.Axis(ticks=False, labels=False, title=None)),
    y=alt.Y('counts', axis=alt.Axis(grid=False)),
    color=alt.Color('ActivityType', legend=alt.Legend(title=None))
    ).properties(
        width=50,
        height=400
    )

    return chart


def bar(data, var):
    counts = data.groupby('ActivityType').sum().reset_index()
    chart = alt.Chart(counts).mark_bar().encode(
    x=alt.X('ActivityType', title = None, sort=alt.EncodingSortField(field=var, order = "descending")),
    y=var
    ).properties(
        width=250,
        height=515
    )
    return chart

def count_num_month(data, var):
    counts = data.groupby(['MonthName', 'ActivityType']).sum().reset_index()

    chart = alt.Chart(counts).mark_bar().encode(
    column=alt.Column(
    'MonthName',
    header=alt.Header(orient='bottom',  labelColor='grey', labelFontSize = 12, labelAngle=0, labelPadding = 28), title = None,
    sort=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ),
    x=alt.X('ActivityType', axis=alt.Axis(ticks=False, labels=False, title=None)),
    y=alt.Y(var, axis=alt.Axis(grid=False)),
    color=alt.Color('ActivityType', legend=alt.Legend(title=None))
    ).properties(
        width=50,
        height=400
    )

    return chart

def scater(data, var1, var2, act_types):
    chart = alt.Chart(data).mark_circle(size=60).encode(
    x=alt.X(var1,axis=alt.Axis( tickSize=100, labelPadding=10)),
    y=alt.Y(var2,axis=alt.Axis( tickSize=100, labelPadding=10)),
    color=alt.Color('ActivityType', legend=alt.Legend(title = None, orient = "top")),
    tooltip=[var1, var2, 'ActivityType']
    ).properties(
        width=700,
        height=400
    ).configure_axis(
    tickCount=10
    )
    return chart


def heat(data, var):
    chart = alt.Chart(data).mark_rect().encode(
    y=alt.Y('month(Date):T',
    title = None),
    x=alt.X('ActivityType', title = None),
    color=var).properties(
    width=425,
    height=500)
    return chart

def box(data, var):
    chart = alt.Chart(data).mark_boxplot(extent='min-max').encode(
    x=alt.X('ActivityType', title = None),
    y=var
    ).properties(
    width=350,
    height=455)
    return chart

def histogram(data, var):
    chart = alt.Chart(data).mark_bar().encode(
    x=alt.X(var, bin=alt.Bin(maxbins=50), title = var),
    y=alt.Y('count():Q', title = "counts"),
    color=alt.Color('ActivityType', legend = alt.Legend(orient = 'top', title = None, columnPadding = 60))
    ).properties(
    width=780,
    height=440
    ).configure_axis(
    tickCount=10
    )
    return chart

def title(txt):
    st.markdown(f'<p style="text-align: center; color:#303030; font-size:40px;">{txt}</p>', unsafe_allow_html=True)

def footnote_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
