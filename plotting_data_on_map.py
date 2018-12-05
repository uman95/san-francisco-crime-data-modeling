#!/usr/bin/env python
# coding: utf-8

# install gmaps first by typing this following line: <br>
# - pip uninstall python-gmaps
# - pip install gmaps
# - jupyter nbextension enable --py --sys-prefix gmaps
# 
# ... then restart any open Jupyter server and close the browser tabs.

# In[1]:




get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import geopandas as gpd
#import pysal as ps
#from pysal.contrib.viz import mapping as maps

from gmplot import gmplot
import pandas as pd
import gmaps
from ipywidgets.embed import embed_minimal_html, embed_snippet
import copy


# In[2]:


crime=pd.read_csv("all/train.csv")
crime.rename(columns={'X': 'longitude', 'Y': 'latitude'}, inplace=True)
crime[['Date', 'Time']]=pd.DataFrame(crime.Dates.str.split(' ',).tolist(), columns=['Date', 'Time'])


# In[3]:


from collections import Counter
list(crime['Category'].drop_duplicates()), Counter(crime['Category'])


#  I will do minibatch of size 10092 <br>
# so the video contain 87 photo in the way that we have 86 minibatch of size  10092 and one with size of  878049

# In[4]:


#names = list(crime['Category'].drop_duplicates())
from matplotlib.colors import rgb2hex
from matplotlib.cm import tab10  ,tab20 , tab20c
# colorss = {year: rgb2hex(tab20c_r(iyear)) for iyear, year in enumerate(names)}
# #colorss
# rgb2hex(tab20c(17)), rgb2hex(tab20c(30))


# In[5]:


#function to display 

import ipywidgets as widgets
 
    
def displayData(df_lat_lon_district, a):
    
    df_lat_lon_district.loc[:,'colors']=copy.deepcopy(df_lat_lon_district['Category'])

    df_lat_lon_district['colors'].replace(   {'WARRANTS': '#1f77b4', 'OTHER OFFENSES': '#aec7e8', 'LARCENY/THEFT': '#ff7f0e',
                                              'VEHICLE THEFT': '#ffbb78', 'VANDALISM': '#2ca02c', 'NON-CRIMINAL': '#98df8a',
                                              'ROBBERY': '#d62728', 'ASSAULT': '#ff9896', 'WEAPON LAWS': '#9467bd', 'BURGLARY': '#c5b0d5',
                                               'SUSPICIOUS OCC': '#8c564b', 'DRUNKENNESS': '#c49c94', 'FORGERY/COUNTERFEITING': '#e377c2',
                                               'DRUG/NARCOTIC': '#f7b6d2', 'STOLEN PROPERTY': '#7f7f7f', 'SECONDARY CODES': '#c7c7c7',
                                               'TRESPASS': '#bcbd22', 'MISSING PERSON': '#dbdb8d', 'FRAUD': '#17becf', 'KIDNAPPING': '#D8BFD8',
                                               'RUNAWAY': '#2E8B57', 'DRIVING UNDER THE INFLUENCE': '#556B2F', 'SEX OFFENSES FORCIBLE': '#9400D3',
                                               'PROSTITUTION': '#FFB6C1', 'DISORDERLY CONDUCT': '#2E8B57', 'ARSON': '#FFFFE0', 'FAMILY OFFENSES': '#FFB6C1',
                                               'LIQUOR LAWS': '#32CD32', 'BRIBERY': '#7FFFD4', 'EMBEZZLEMENT': '#ff8000', 'SUICIDE': '#9edae5', 'LOITERING': '#ff4000',
                                               'SEX OFFENSES NON FORCIBLE': '#D8BFD8', 'EXTORTION': '#008080', 'GAMBLING': '#6B8E23', 'BAD CHECKS': '#2E8B57',
                                               'TREA': '#6B8E23', 'RECOVERED VEHICLE': '#ff8000', 'PORNOGRAPHY/OBSCENE MAT': '#ADFF2F'}, inplace=True)
            



    gmaps.configure(api_key='AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY')

    fig = gmaps.figure(map_type='ROADMAP')#, display_toolbar=False)

    data_new=copy.deepcopy(df_lat_lon_district[['latitude', 'longitude','colors','Category']].drop_duplicates())
    locations = data_new[['latitude', 'longitude']][a: a+1000]  # [(51.5, 0.1), (51.7, 0.2), (51.4, -0.2), (51.49, 0.1)]
    colorss = list(data_new['colors'][a: a+1000])
    names = list(data_new['Category'][a: a+1000])

    symbolmap_layer = gmaps.symbol_layer(locations, hover_text=names,
                                          fill_color=colorss, stroke_color=colorss)#, display_info_box=True)
    fig.add_layer(symbolmap_layer)

    embed_minimal_html(fp='images/img_%s.html'%a, views=[fig], title='crime')
    return fig

    


# In[11]:


DATA_NORTHERN =crime[(crime['PdDistrict']=='TARAVAL') & (crime['Category']!='OTHER OFFENSES')][['latitude','longitude', 'Category']]
len(DATA_NORTHERN.drop_duplicates())


# In[23]:


from time import sleep
DATA_TARAVAL =crime[(crime['PdDistrict']=='TARAVAL') & (crime['Category']!='OTHER OFFENSES')][['latitude','longitude', 'Category']]

DATA_TARAVAL=DATA_TARAVAL.drop_duplicates()
#displayData(DATA_)
from IPython.display import IFrame

for i in range(0,len(DATA_TARAVAL)-1000,2320):
    
    display(displayData(DATA_TARAVAL,i))
    #display(IFrame("images/img_%s.html"%i, width=350, height=315))

    sleep(3)


# In[24]:


import imageio
import glob ## no installation require
images=[]
filenames=glob.glob('images/TARAVAL/*.png')
for c in filenames:
    images.append(imageio.imread(c))
kargs={'duration': 0.4}

imageio.mimsave('images/TARAVAL/TARAVAL.gif', images, **kargs)


# In[ ]:




