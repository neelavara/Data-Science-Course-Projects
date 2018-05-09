
# coding: utf-8

# In[148]:


import pandas as pd
import csv
import re


# In[149]:


path = "C:\Users\Owner\Desktop\Merged.csv"
merged = pd.read_csv(path)


# In[150]:


allcusinesList = []
allpinCode = []
takeOutYes = 0
takeOutNo = 0

#Cusine list
for item in merged.Cuisines:
    #miniList = item.split(';')
    #for one in miniList:
    allcusinesList.extend(item.split(';'))
allcusinesList = map(str.strip, allcusinesList)
allcusinesList = map(str.lower,allcusinesList)

#Take out count
for val in merged.Take_Out:
    if val == 'Yes':
        takeOutYes +=1
    else:
        takeOutNo +=1
#Zip Code
for pin in merged.Address:
    code = pin.partition("Ny")[2]
    code = code.split("-")[0]
    allpinCode.append(code.strip())


# In[151]:


count = {}
for cusine in allcusinesList:
    if cusine in count.keys():
        count[cusine] +=1
    else:
        count[cusine] = 1
pinCode = {}
for pin in allpinCode:
    if pin in pinCode.keys():
        pinCode[pin] +=1
    else:
        pinCode[pin] = 1


# In[152]:


with open('Cusine Count.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Type of Cusine','Number of Restaurants that have this Cusine'])
    for key, value in count.items():
        writer.writerow([key, value])


# In[153]:


with open('Pin Code.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Pin Code','Number of Restaurants in that Area'])
    for key, value in pinCode.items():
        writer.writerow([key, value])


# In[154]:


print("Take out option available: " + str(takeOutYes))


# In[155]:


print("Take out option not available: " + str(takeOutNo))


# In[182]:


#res = merged.groupby(['Name'])
#res = merged.groupby('Shake Shake').count()
a = merged['Name'].value_counts(ascending=False)


# In[187]:


a.to_csv('chain.csv')

