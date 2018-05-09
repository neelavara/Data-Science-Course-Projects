
# coding: utf-8

# # Project Stage 3: Entity Matching#
# 
# The entity we performed our match was on a set of restaurants in New York City. The data for the tables was taken from two different web sources, namely, Tripadvisor and Yelp.
# 
# We used the py_entitymatching package to help in this process.

# In[241]:


# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd


# ## Reading in Input Tables##
# 
# We read the CSV files and set 'ID' as the key attribute.
# 
# Input table A corresponds to data from TripAdvisor, and input table B corresponds to data from Yelp.

# In[242]:


path_A = 'C:\\Users\\bharg\\Documents\\TripAdvisor_Restaurants.csv'
path_B = 'C:\\Users\\bharg\\Documents\\Yelp_Restaurants.csv'
#path_A = 'C:\\Users\\Aribhit\\TripAdvisor_Restaurants.csv'
#path_B = 'C:\\Users\\Aribhit\\Yelp_Restaurants.csv'


# In[243]:


A = em.read_csv_metadata(path_A, key='Id', encoding = 'cp1252')


# In[244]:


B = em.read_csv_metadata(path_B, key='Id', encoding = 'cp1252')


# # Data Pre-processing#
# 
# Since there was a lot of variance in how *Address* attribute was defined, created a new attribute called *Street* that is extracted from *Address* (by considering the part of the string till the state appears).
# 
# Also converted every string type attribute to lower case.

# In[245]:


A['Street'] = A.apply(lambda row : row['Address'][0:row['Address'].find('New')],axis=1)
A['Name']=A['Name'].str.lower()
A['Street']=A['Street'].str.lower()
A['Address']=A['Address'].str.lower()


# In[246]:


def cleaning(row) :
    for string in ['New','Jersey','NY','NJ']:
        index = row['Address'].find(string)
        if index == -1 :
            continue
        return row['Address'][0:index]
    return row['Address']
    
B['Street'] = B.apply(cleaning,axis =1)
B['Name']=B['Name'].str.lower()
B['Street']=B['Street'].str.lower()
B['Address']=B['Address'].str.lower()


# Deleting *Phone* attribute in case it might act as an unique ID and game the system

# In[247]:


#A1 = A.copy(deep=True)
#B1 = B.copy(deep=True)
#del A['Phone']
#del B['Phone']


# ## Applying the Blocker##
# 
# We have used the combination of two blockers: 
# One blocks based on *Name* of the restaurant (Jaccard Measure with 3 grams with a constraint 0.3) and *Street* (Jaccard Measure with 3 grams with a constraint 0.3). 
# Next blocker is only on the *Street* attribute(Jaccard Measure with 3 grams with a constraint 0.6).
# 
# We use these blockers and then combine the results of two different blockers using union for the following reasons.
# *Street* (from *Address*) only because it can capture some pairs where names are same but differ by a new word. Ex. (alfa ristorante, alfa). The constraint is higher - 0.6
# *Name* only to capture restaurants that have similar names (constraint is 0.3). But added *Street* based rule on top of that to eliminate chain restaurants with multiple branches at different locations (eg: Shake shacks at Manhattan, Shake Shacks at Brooklyn). The constraint is lower threshold in this case compared to the earlier blocker.
# 
# First, get all possible features for blocking.

# In[248]:


block_f = em.get_features_for_blocking(A, B, validate_inferred_attr_types=False)


# First rule-based blocker uses *Name* and *Street* attributes

# In[249]:


rb = em.RuleBasedBlocker()
ab = em.AttrEquivalenceBlocker()
rb.add_rule(['Name_Name_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.5'], block_f)
rb.add_rule(['Street_Street_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.3'], block_f)


# In[250]:


C = rb.block_tables(A, B, l_output_attrs=['Name', 'Street', 'Address','Cuisines','Take Out','Phone',
                                          'Saturday Opening time','Saturday Closing time','Sunday Opening time',
                                          'Sunday Closing time'], 
                    r_output_attrs=['Name', 'Street', 'Address','Cuisines','Take Out','Phone',
                                          'Saturday Opening time','Saturday Closing time','Sunday Opening time',
                                          'Sunday Closing time'], show_progress=True)


# Second rule-based blocker uses only *Street* attribute

# In[251]:


rb2 = em.RuleBasedBlocker()
rb2.add_rule(['Street_Street_jac_qgm_3_qgm_3(ltuple, rtuple) < 0.6'], block_f)
E = rb2.block_tables(A, B, l_output_attrs=['Name', 'Street', 'Address','Cuisines','Take Out','Phone',
                                          'Saturday Opening time','Saturday Closing time','Sunday Opening time',
                                          'Sunday Closing time'], 
                     r_output_attrs=['Name', 'Street', 'Address','Cuisines','Take Out','Phone',
                                          'Saturday Opening time','Saturday Closing time','Sunday Opening time',
                                          'Sunday Closing time'], n_jobs=-1,show_progress=True)


# Combining blocker1 and blocker2 results to get candidate set C (which is named F in our code).

# In[252]:


F = em.combine_blocker_outputs_via_union([C, E])


# Running debugger to see if F is good. 41/50 outputs of debugger are bad matches.Therefore we are proceeding with the above 
# blocker

# In[14]:


dbg = em.debug_blocker(F, A, B, output_size=50)
dbg.head()


# In[253]:


F.to_csv("F.csv",index=False,encoding = 'cp1252')


# In[254]:


F.shape


# In[255]:


F.head()


# Taking a sample of 600 tuples from the output, and then we label this sample manually.

# In[16]:


S = em.sample_table(F, 600)
S.to_csv('Sample.csv',encoding = 'cp1252')


# ## Reading the Labelled Sample##
# Loading the labeled data table, which is present in a file called 'Labelled_Sample_v2.csv'

# In[256]:


L = em.read_csv_metadata("Labelled_Sample_v2.csv", key='_id', encoding = 'cp1252',                         ltable=A, rtable=B,fk_ltable='ltable_Id', fk_rtable='rtable_Id')


# Deleting *Phone* attribute again, because it can help determine matches trivially.

# In[14]:


del L['ltable_Phone']
del L['rtable_Phone']


# ## Splitting the Labelled Set##
# 
# Splitting the labelled set into training and test set, by putting half the tuple pairs in each.<br>
# The development set is called I<br>
# The evaluation set is called J

# In[257]:


IJ = em.split_train_test(L, train_proportion=0.5, random_state=0)
I = IJ['train']
J = IJ['test']
I.to_csv('I.csv',encoding = 'cp1252')
J.to_csv('J.csv',encoding = 'cp1252')


# In[258]:


J.head()


# ## Creating ML-matchers##
# 
# Initiating 6 different classifiers (Decision Tree, Random Forest, SVM, Naive Bayes, Logistic Regression, Linear Regression) and then, cross validating them on I set.

# In[259]:


dt = em.DTMatcher(name='DecisionTree', random_state=0)
rf = em.RFMatcher(name='RF', random_state=0)
svm = em.SVMMatcher(name='SVM', random_state=0)
nb = em.NBMatcher(name ='NaiveBayes')
lg = em.LogRegMatcher(name='LogReg', random_state=0)
ln = em.LinRegMatcher(name='LinReg')


# ## Selecting Best Matcher ##
# 
# First, we obtain all the features we could use for matching. Ft is our feature table

# In[260]:


Ft = em.get_features_for_matching(A, B, validate_inferred_attr_types=False)


# Use the system to generate feature vectors from set I. This is called set H

# In[261]:


H = em.extract_feature_vecs(I, 
                            feature_table=Ft, 
                            attrs_after='label',
                            show_progress=False)


# Perform matches and display results below (after performing cross-validation)

# In[262]:


H = em.impute_table(H, 
                exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
                strategy='mean')


# In[161]:


result = em.select_matcher([dt, rf, svm, ln, lg,nb], table=H, 
        exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'],
        k=5,
        target_attr='label', metric_to_select_matcher='f1', random_state=0)
result['cv_stats']


# Picking Random Forest as it has the highest average F1 score. We are not adding any rule based matchers as the precision,recall and F1 scores are already above the required thresholds.

# ## Evaluating Best Matcher##
# 
# As we picked Random Forest as the best matcher, now we apply it on the evaluation set (set J; defined earlier) to find how well it performs.
# 
# Create a new Random Forest matcher and train it on set H (feature table obtained from set I):

# In[263]:


rf = em.RFMatcher(name='RF', random_state=0)


# In[264]:


rf.fit(table=H, 
       exclude_attrs=['_id', 'ltable_Id', 'rtable_Id', 'label'], 
       target_attr='label')


# In[265]:


Test_Ft = em.read_csv_metadata("F.csv", key='_id', encoding = 'cp1252',                         ltable=A, rtable=B,fk_ltable='ltable_Id', fk_rtable='rtable_Id')


# Extracting features from set J:

# In[266]:


Test_Ft = em.extract_feature_vecs(Test_Ft, feature_table=Ft,
                             show_progress=False)


# In[267]:


Test_Ft = Test_Ft.dropna(axis =0,how ='any')
Test_Ft.to_csv("Test_Ft.csv",index=False)


# In[268]:


Test_Ft = em.read_csv_metadata("Test_Ft.csv", key='_id', encoding = 'cp1252',                         ltable=A, rtable=B,fk_ltable='ltable_Id', fk_rtable='rtable_Id')


# In[269]:


Test_Ft.head()


# Computing predictions on set J:

# In[270]:


predictions = rf.predict(table=Test_Ft, exclude_attrs=['_id', 'ltable_Id', 'rtable_Id'], 
              append=True, target_attr='predicted', inplace=False, return_probs=True,
                        probs_attr='proba')


# In[271]:


predictions[['_id', 'ltable_Id', 'rtable_Id', 'predicted', 'proba']]


# In[272]:


ids = predictions['_id'][predictions['predicted']==1]


# In[273]:


ids = ids.tolist()


# In[274]:


F = em.read_csv_metadata("F.csv", key='_id', encoding = 'cp1252',                         ltable=A, rtable=B,fk_ltable='ltable_Id', fk_rtable='rtable_Id')


# In[275]:


F.shape


# In[276]:


F = F[F['_id'].isin(ids)]


# In[277]:


F.shape


# In[298]:


F.head(200)


# In[ ]:


del F['_id']
del F['ltable_Id']
del F['rtable_Id']


# # Project Stage 4 - Data Merging
# Please find the code for the Data Merging part of Stage 4 below.

# In[280]:


p4 = pd.DataFrame(columns=['Name', 'Address','Cuisines','Take Out','Phone',
                                          'Saturday Opening time','Saturday Closing time','Sunday Opening time',
                                          'Sunday Closing time'])


# In[309]:



def name_merger(row) :
    return row['ltable_Name'][:1].upper()+row['ltable_Name'][1:] if len(row['ltable_Name']) >= len(row['rtable_Name'])    else row['rtable_Name'][:1].upper()+row['rtable_Name'][1:] 

def address_merger(row) :
    return row['ltable_Address'].title() if len(row['ltable_Address']) >= len(row['rtable_Address'])     else row['rtable_Address'].title()

def cuisine_merger(row):
    result = set()
    union1 = row['ltable_Cuisines'].split(';')
    union1 = [x.lower() for x in union1]
    union2 = row['rtable_Cuisines'].split(';')
    union2 = [x.lower() for x in union2]
    if len(union1)>=len(union2):
        result.update(union1)
    else:
        result.update(union2)
    return "; ".join([x.title() for x in list(result)])

def takeout_merger(row) :
    if row['ltable_Take Out'] == 'Yes' or row['rtable_Take Out'] == 'Yes' :
        return 'Yes'
    else :
        return 'No'

def phone_merger(row) :
    return row['ltable_Phone'] if len(row['ltable_Phone']) >= len(row['rtable_Phone'])     else row['rtable_Phone']    
    
def sat_open_merger(row) :
    if row['ltable_Saturday Opening time'].lower() == 'nan' :
        return row['rtable_Saturday Opening time']
    if row['rtable_Saturday Opening time'].lower() == 'nan' :
        return row['ltable_Saturday Opening time']
    if row['rtable_Saturday Opening time'].lower() == 'closed' or row['ltable_Saturday Opening time'].lower() == 'closed':
        return 'Closed'
    return row['ltable_Saturday Opening time'] if len(row['ltable_Saturday Opening time']) <= len(row['rtable_Saturday Opening time'])     else row['rtable_Saturday Opening time']  

def sat_close_merger(row) :
    if row['ltable_Saturday Closing time'].lower() == 'nan' :
        return row['rtable_Saturday Closing time']
    if row['rtable_Saturday Closing time'].lower() == 'nan' :
        return row['ltable_Saturday Closing time']
    if row['rtable_Saturday Closing time'].lower() == 'closed' or row['ltable_Saturday Closing time'].lower() == 'closed':
        return 'Closed'
    return row['ltable_Saturday Closing time'] if len(row['ltable_Saturday Closing time']) <= len(row['rtable_Saturday Closing time'])     else row['rtable_Saturday Closing time']  

def sun_open_merger(row) :
    if row['ltable_Sunday Opening time'].lower() == 'nan' :
        return row['rtable_Sunday Opening time']
    if row['rtable_Sunday Opening time'].lower() == 'nan' :
        return row['ltable_Sunday Opening time']
    if row['rtable_Sunday Opening time'].lower() == 'closed' or row['ltable_Sunday Opening time'].lower() == 'closed':
        return 'Closed'
    return row['ltable_Sunday Opening time'] if len(row['ltable_Sunday Opening time']) <= len(row['rtable_Sunday Opening time'])     else row['rtable_Sunday Opening time']  
    
def sun_close_merger(row):
    if row['ltable_Sunday Closing time'].lower() == 'nan' :
        return row['rtable_Sunday Closing time']
    if row['rtable_Sunday Closing time'].lower() == 'nan' :
        return row['ltable_Sunday Closing time']
    if row['rtable_Sunday Closing time'].lower() == 'closed' or row['ltable_Sunday Closing time'].lower() == 'closed':
        return 'Closed'
    return row['ltable_Sunday Closing time'] if len(row['ltable_Sunday Closing time']) <= len(row['rtable_Sunday Closing time'])     else row['rtable_Sunday Closing time']  


# In[310]:


p4['Name'] = F.apply(name_merger,axis =1)
p4['Address'] = F.apply(address_merger,axis =1)
p4['Cuisines'] = F.apply(cuisine_merger,axis =1)
p4['Take Out'] = F.apply(takeout_merger,axis =1)
p4['Phone'] = F.apply(phone_merger,axis =1)
p4['Saturday Opening time'] = F.apply(sat_open_merger,axis =1)
p4['Saturday Closing time'] = F.apply(sat_close_merger,axis =1)
p4['Sunday Opening time'] = F.apply(sun_open_merger,axis =1)
p4['Sunday Closing time'] = F.apply(sun_close_merger,axis =1)


# In[311]:


p4.head(100)


# In[312]:


p4.to_csv("E.csv",index = False,encoding = 'cp1252')

