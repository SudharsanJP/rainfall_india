
import pandas as pd
import streamlit as st
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

st.title(':orange[💮rainfall in india - Mini ML Project]🌞')

#)reading the dataset
df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\capstone project\rainfall\rainfall in india 1901-2015.csv")
st.subheader("\n:green[1. dataset analysis🌝]\n")
if (st.checkbox("original data")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.1 original dataframe]\n")
    data = df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:purple'))
#) check for the null values
df.isna().sum()

#) filling the null values
df = df.ffill()
df = df.bfill()

#) check for the null values
df.isna().sum()

#) year and JAN dataframe
jan_df = df[['SUBDIVISION','YEAR','JAN']]


#) making JAN data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("JAN")

#) converting list into column
jan_df['month'] = month_list


#) chanhing the column name JAN --> vol
jan_df.rename(columns = {'JAN':'vol'}, inplace = True)


#) year and FEB dataframe
feb_df = df[['SUBDIVISION','YEAR','FEB']]


#) making FEB data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("FEB")

#) converting list into column
feb_df['month'] = month_list


#) chanhing the column name FEB --> vol
feb_df.rename(columns = {'FEB':'vol'}, inplace = True)


#) year and MAR dataframe
mar_df = df[['SUBDIVISION','YEAR','MAR']]

#) making MAR data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("MAR")

#) converting list into column
mar_df['month'] = month_list


#) chanhing the column name MAR --> vol
mar_df.rename(columns = {'MAR':'vol'}, inplace = True)


#) year and APR dataframe
apr_df = df[['SUBDIVISION','YEAR','APR']]


#) making APR data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("APR")

#) converting list into column
apr_df['month'] = month_list


#) chanhing the column name APR --> vol
apr_df.rename(columns = {'APR':'vol'}, inplace = True)


#) year and MAY dataframe
may_df = df[['SUBDIVISION','YEAR','MAY']]


#) making MAY data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("MAY")

#) converting list into column
may_df['month'] = month_list


#) chanhing the column name MAY --> vol
may_df.rename(columns = {'MAY':'vol'}, inplace = True)


#) year and JUN dataframe
jun_df = df[['SUBDIVISION','YEAR','JUN']]


#) making JUN data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("JUN")

#) converting list into column
jun_df['month'] = month_list


#) chanhing the column name JUN --> vol
jun_df.rename(columns = {'JUN':'vol'}, inplace = True)


#) year and JUL dataframe
jul_df = df[['SUBDIVISION','YEAR','JUL']]


#) making JUL data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("JUL")


#) converting list into column
jul_df['month'] = month_list


#) chanhing the column name JUL --> vol
jul_df.rename(columns = {'JUL':'vol'}, inplace = True)


#) year and AUG dataframe
aug_df = df[['SUBDIVISION','YEAR','AUG']]


#) making AUG data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("AUG")


#) converting list into column
aug_df['month'] = month_list


#) chanhing the column name AUG --> vol
aug_df.rename(columns = {'AUG':'vol'}, inplace = True)


#) year and SEP dataframe
sep_df = df[['SUBDIVISION','YEAR','SEP']]


#) making SEP data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("SEP")


#) converting list into column
sep_df['month'] = month_list


#) chanhing the column name SEP --> vol
sep_df.rename(columns = {'SEP':'vol'}, inplace = True)


#) year and OCT dataframe
oct_df = df[['SUBDIVISION','YEAR','OCT']]


#) making OCT data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("OCT")


#) converting list into column
oct_df['month'] = month_list


#) changing the column name OCT --> vol
oct_df.rename(columns = {'OCT':'vol'}, inplace = True)


#) year and NOV dataframe
nov_df = df[['SUBDIVISION','YEAR','NOV']]


#) making NOV data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("NOV")


#) converting list into column
nov_df['month'] = month_list


#) changing the column name NOV --> vol
nov_df.rename(columns = {'NOV':'vol'}, inplace = True)


#) year and DEC dataframe
dec_df = df[['SUBDIVISION','YEAR','DEC']]


#) making DEC data in a column
month_list = []
length = 10
for name in range(0,4116):
    month_list.append("DEC")


#) converting list into column
dec_df['month'] = month_list


#) changing the column name DEC --> vol
dec_df.rename(columns = {'DEC':'vol'}, inplace = True)

#) concatenation of 12 dataframes
new_df = pd.concat([jan_df, feb_df,mar_df,apr_df,
               may_df,jun_df,jul_df,aug_df,
               oct_df,sep_df,nov_df,dec_df], axis=0, ignore_index=True)

#) to change the order of column 
new_df = new_df.iloc[:,[0,1,3,2]]

if (st.checkbox("data postprocessing")):
    #)showing original dataframe
    st.markdown("\n#### :red[1.2 data postprocessing]\n")
    data = new_df.head(5)
    st.dataframe(data.style.applymap(lambda x: 'color:green'))

#new_df['month'].unique()

#)scatterplot
if (st.checkbox("scatterplot")):
    st.markdown("#### :green[1.3 scatterplot]")
    #)scatter plot
    fig = px.scatter(
    new_df,
    x="YEAR",
    y="vol",
    color='month',
    log_x=True,
    size_max=60,
    )
    tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
    with tab1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with tab2:
        st.plotly_chart(fig, theme=None, use_container_width=True)

#) andhaman rain data
andhaman_df = new_df[new_df['SUBDIVISION'] == 'ANDAMAN & NICOBAR ISLANDS']
andhaman_rain = andhaman_df['vol'].mean()

#) arunachal rain data
arunachal_df = new_df[new_df['SUBDIVISION'] == 'ARUNACHAL PRADESH']
arunachal_rain = arunachal_df['vol'].mean()

#) assam rain data
assam_df = new_df[new_df['SUBDIVISION'] == 'ASSAM & MEGHALAYA']
assam_rain = assam_df['vol'].mean()

#) naga rain data
naga_df = new_df[new_df['SUBDIVISION'] == 'NAGA MANI MIZO TRIPURA']
naga_rain = naga_df['vol'].mean()

#) himalaya_rain rain data
himalaya_df = new_df[new_df['SUBDIVISION'] == 'SUB HIMALAYAN WEST BENGAL & SIKKIM']
himalaya_rain = himalaya_df['vol'].mean()
#)scatterplot
if (st.checkbox("mean rain statewise")):
    st.markdown("#### :green[1.4 mean rain statewise]")
    st.write(f"himalaya rain: {himalaya_rain}")
    st.write(f"naga rain: {naga_rain}")
    st.write(f"assam rain: {assam_rain}")
    st.write(f"arunachal rain: {arunachal_rain}")
    st.write(f"andhaman rain: {andhaman_rain}")

#) gangestic_rain rain data
gangestic_df = new_df[new_df['SUBDIVISION'] == 'GANGETIC WEST BENGAL']
gangestic_rain = gangestic_df['vol'].mean()
#print(f"gangestic rain: {gangestic_rain}")

#) gangestic_rain rain data
jarkhand_df = new_df[new_df['SUBDIVISION'] == 'JHARKHAND']
jarkhand_rain = jarkhand_df['vol'].mean()
#print(f"jarkhand rain: {jarkhand_rain}")

#) bihar_rain rain data
bihar_df = new_df[new_df['SUBDIVISION'] == 'BIHAR']
bihar_rain = bihar_df['vol'].mean()
#print(f"bihar rain: {bihar_rain}")

#) bihar_rain rain data
eastup_df = new_df[new_df['SUBDIVISION'] == 'EAST UTTAR PRADESH']
eastup_rain = eastup_df['vol'].mean()
#print(f"east UP rain: {eastup_rain}")

#) orissa rain data
orissa_df = new_df[new_df['SUBDIVISION'] == 'ORISSA']
orissa_rain = orissa_df['vol'].mean()
#print(f"orissa rain: {orissa_rain}")

#) westup rain data
westup_df = new_df[new_df['SUBDIVISION'] == 'WEST UTTAR PRADESH']
westup_rain = westup_df['vol'].mean()
#print(f"west UP rain: {westup_rain}")

#) uk rain data
uk_df = new_df[new_df['SUBDIVISION'] == 'UTTARAKHAND']
uk_rain = uk_df['vol'].mean()
print(f"uk rain: {uk_rain}")

#) haryana delhi and chandigarh rain data
har_df = new_df[new_df['SUBDIVISION'] == 'HARYANA DELHI & CHANDIGARH']
har_rain = har_df['vol'].mean()
#print(f"haryana delhi and chandigarh rain: {har_rain}")

#) punjab rain data
punjab_df = new_df[new_df['SUBDIVISION'] == 'PUNJAB']
punjab_rain = punjab_df['vol'].mean()
#print(f"orissa rain: {orissa_rain}")

#) himachal rain data
himachal_df = new_df[new_df['SUBDIVISION'] == 'HIMACHAL PRADESH']
himachal_rain = himachal_df['vol'].mean()
#print(f"himachal rain: {himachal_rain}")

#) JAMMU & KASHMIR rain data
jk_df = new_df[new_df['SUBDIVISION'] == 'JAMMU & KASHMIR']
jk_rain = jk_df['vol'].mean()
#print(f"JAMMU & KASHMIR rain: {orissa_rain}")

#) west rajasthan rain data
w_rajasthan_df = new_df[new_df['SUBDIVISION'] == 'WEST RAJASTHAN']
w_rajasthan_rain = w_rajasthan_df['vol'].mean()
#print(f"west rajasthan rain: {w_rajasthan_rain}")

#) east rajasthan rain data
e_rajasthan_df = new_df[new_df['SUBDIVISION'] == 'EAST RAJASTHAN']
e_rajasthan_rain = e_rajasthan_df['vol'].mean()
#print(f"eest rajasthan rain: {e_rajasthan_rain}")

#) west mp rain data
wmp_df = new_df[new_df['SUBDIVISION'] == 'WEST MADHYA PRADESH']
wmp_rain = wmp_df['vol'].mean()
#print(f"west mp rain: {wmp_rain}")

#) east mp rain data
emp_df = new_df[new_df['SUBDIVISION'] == 'EAST MADHYA PRADESH']
emp_rain = emp_df['vol'].mean()
#print(f"west mp rain: {emp_rain}")

#) guj rain data
guj_df = new_df[new_df['SUBDIVISION'] == 'GUJARAT REGION']
guj_rain = guj_df['vol'].mean()
#print(f"gujarat rain: {guj_rain}")

#) kutch rain data
kutch_df = new_df[new_df['SUBDIVISION'] == 'SAURASHTRA & KUTCH']
kutch_rain = kutch_df['vol'].mean()
#print(f"kutch and saurashtra rain: {kutch_rain}")

#) konkan and goa rain data
goa_df = new_df[new_df['SUBDIVISION'] == 'KONKAN & GOA']
goa_rain = goa_df['vol'].mean()
#print(f"konkan and goa rain: {goa_rain}")

#) madhya maharashtra rain data
mmah_df = new_df[new_df['SUBDIVISION'] == 'MADHYA MAHARASHTRA']
mmah_rain = mmah_df['vol'].mean()
#print(f"MADHYA MAHARASHTRA rain: {mmah_rain}")

#) mathathwadha rain data
mathathwadha_df = new_df[new_df['SUBDIVISION'] == 'MATATHWADA']
mathathwadha_rain = mathathwadha_df['vol'].mean()
#print(f"MATATHWADA rain: {mathathwadha_rain}")

#) vidharbha rain data
vidharbha_df = new_df[new_df['SUBDIVISION'] == 'VIDARBHA']
vidharbha_rain = vidharbha_df['vol'].mean()
#print(f"vidharbha and goa rain: {vidharbha_rain}")

#) chatisgarh rain data
hatisgarh_df = new_df[new_df['SUBDIVISION'] == 'CHHATTISGARH']
hatisgarh_rain = hatisgarh_df['vol'].mean()
#print(f"hatisgarh rain: {hatisgarh_rain}")

#) coastal ap rain data
coastalap_df = new_df[new_df['SUBDIVISION'] == 'COASTAL ANDHRA PRADESH']
coastalap_rain = coastalap_df['vol'].mean()
#print(f"coastal ap rain: {coastalap_rain}")

#) telungana rain data
telungana_df = new_df[new_df['SUBDIVISION'] == 'TELANGANA']
telungana_rain = telungana_df['vol'].mean()
#print(f"telungana rain: {telungana_rain}")

#) rayalseema rain data
rayalseema_df = new_df[new_df['SUBDIVISION'] == 'RAYALSEEMA']
rayalseema_rain = rayalseema_df['vol'].mean()
#print(f"rayalseema rain: {rayalseema_rain}")

#) tamil rain data
tamil_df = new_df[new_df['SUBDIVISION'] == 'TAMIL NADU']
tamil_rain = goa_df['vol'].mean()
#print(f"tamindau rain: {tamil_rain}")

#) coastal kannada rain data
ckannada_df = new_df[new_df['SUBDIVISION'] == 'COASTAL KARNATAKA']
ckannada_rain = ckannada_df['vol'].mean()
#print(f"coastal kannada rain: {ckannada_rain}")
 
#) north interior knnada rain data
n_ikannada_df = new_df[new_df['SUBDIVISION'] == 'NORTH INTERIOR KARNATAKA']
n_ikannada_rain = n_ikannada_df['vol'].mean()
#print(f"north interior knnada rain: {n_ikannada_rain}")

#) south interior knnada rain data
s_ikannada_df = new_df[new_df['SUBDIVISION'] == 'SOUTH INTERIOR KARNATAKA']
s_ikannada_rain = s_ikannada_df['vol'].mean()
#print(f"north interior knnada rain: {s_ikannada_rain}")

#) kerala rain data
kerala_df = new_df[new_df['SUBDIVISION'] == 'KERALA']
kerala_rain = kerala_df['vol'].mean()
#print(f"kerala rain: {kerala_rain}")

#) lakshadweep rain data
lakshadweep_df = new_df[new_df['SUBDIVISION'] == 'LAKSHADWEEP']
lakshadweep_rain = lakshadweep_df['vol'].mean()
#print(f"lakshadweep rain: {lakshadweep_rain}")

#new_df['SUBDIVISION'].unique()

import numpy as np
#) converting df to list
orissa_year_list =orissa_df['YEAR'].tolist()
orissa_list_vol = orissa_df['vol'].tolist()

#) converting list to np
np_orissa_year = np.array(orissa_year_list)
np_orissa_vol = np.array(orissa_list_vol)

if (st.checkbox("scatterplot2")):
    #)showing original dataframe
    st.markdown("\n#### :blue[1.5 orissa rain scatterplot]\n")
    fig,ax = plt.subplots(figsize=(15,8))
    ax.scatter(np_orissa_year,np_orissa_vol,color = 'yellow')
    st.pyplot(fig)

#new_df['SUBDIVISION'].unique()

#) ml prediction
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

new_df = pd.get_dummies(new_df,['SUBDIVISION','month'])
X = new_df.drop(['vol'],axis=1)
y = new_df['vol']

model = LinearRegression()
model.fit(X,y)
new_df['prediction'] = model.predict(X)
mse = mean_squared_error(new_df['vol'],new_df['prediction'])

#) machine learning models
st.subheader(":red[2.machine learning models]")
if (st.checkbox("linear regression model")):
    st.markdown("#### :violet[2.1 linear regression model]")
    st.write(mse)

#)ridge
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Ridge()
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

if (st.checkbox("ridge")):
    st.markdown("#### :violet[2.2 ridge]")
    st.success("********Train data*********")
    st.write(mean_squared_error(y_train,train_pred))
    
    st.error("********Test data*********")
    st.write(mean_squared_error(y_test,test_pred))