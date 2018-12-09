import pandas as pd
import numpy as np
import matplotlib as plt
pd.set_option('display.max_columns', None)
df=pd.read_csv('train_HK6lq50.csv')

def train_data_preprocess(df,train,test):
    
    df['trainee_engagement_rating'].fillna(value=1.0,inplace=True)   

    df['isage_null']=0
    df.isage_null[df.age.isnull()]=1                                
    df['age'].fillna(value=0,inplace=True)        

    #new cols actual_programs_enrolled and total_test_taken
    total=train.append(test)
    unique_trainee=pd.DataFrame(total.trainee_id.value_counts())
    unique_trainee['trainee_id']=unique_trainee.index

    value=[]
    for i in unique_trainee.trainee_id:
        value.append(len(total[total.trainee_id==i].program_id.unique()))
    unique_trainee['actual_programs_enrolled']=value
    dic1=dict(zip(unique_trainee['trainee_id'],unique_trainee['actual_programs_enrolled']))
    df['actual_programs_enrolled']=df['trainee_id'].map(dic1).astype(int)

    value=[]
    for i in unique_trainee.trainee_id:
        value.append(len(total[total.trainee_id==i].test_id.unique()))
    unique_trainee['total_test_taken']=value
    dic2=dict(zip(unique_trainee['trainee_id'],unique_trainee['total_test_taken']))
    df['total_test_taken']=df['trainee_id'].map(dic2).astype(int)


    #new col total_trainee_in_each_test
    unique_test=pd.DataFrame(total.test_id.value_counts())
    unique_test['test_id']=unique_test.index

    value=[]
    for i in unique_test.test_id:
        value.append(len(total[total.test_id==i].trainee_id.unique()))
    unique_test['total_trainee_in_each_test']=value
    dic3=dict(zip(unique_test['test_id'],unique_test['total_trainee_in_each_test']))
    df['total_trainee_in_each_test']=df['test_id'].map(dic3).astype(int)


    #LABEL ENCODING
    
    test_type=sorted(df['test_type'].unique())
    test_type_mapping=dict(zip(test_type,range(1,len(test_type)+1)))
    df['test_type_val']=df['test_type'].map(test_type_mapping).astype(int)
    df.drop('test_type',axis=1,inplace=True)

    program_type=sorted(df['program_type'].unique())
    program_type_mapping=dict(zip(program_type,range(1,len(program_type)+1)))
    df['program_type_val']=df['program_type'].map(program_type_mapping).astype(int)
    df.drop('program_type',axis=1,inplace=True)

    program_id=sorted(df['program_id'].unique())
    program_id_mapping=dict(zip(program_id,range(1,len(program_id)+1)))
    df['program_id_val']=df['program_id'].map(program_id_mapping).astype(int)
    #df.drop('program_id',axis=1,inplace=True)

    difficulty_level=['easy','intermediate','hard','vary hard']
    difficulty_level_mapping=dict(zip(difficulty_level,range(1,len(difficulty_level)+1)))
    df['difficulty_level_val']=df['difficulty_level'].map(difficulty_level_mapping).astype(int)
    df.drop('difficulty_level',axis=1,inplace=True)

    education=['No Qualification','High School Diploma','Matriculation','Bachelors','Masters']
    educationmapping=dict(zip(education,range(1,len(education)+1)))
    df['education_val']=df['education'].map(educationmapping).astype(int)
    df.drop('education',axis=1,inplace=True)

    is_handicapped=sorted(df['is_handicapped'].unique())
    is_handicappedmapping=dict(zip(is_handicapped,range(1,len(is_handicapped)+1)))
    df['is_handicapped_val']=df['is_handicapped'].map(is_handicappedmapping).astype(int)
    df.drop('is_handicapped',axis=1,inplace=True)

    #creating new program_id group based on is_pass percentage
    df['new_program_id_group']=pd.DataFrame(df['program_id'])
    df.loc[(df.new_program_id_group=='X_1')|(df.new_program_id_group=='X_3'),'new_program_id_group']=1
    df.loc[(df.new_program_id_group=='Y_1')|(df.new_program_id_group=='Y_2')|(df.new_program_id_group=='Y_3')|(df.new_program_id_group=='Y_4')|(df.new_program_id_group=='X_2'),'new_program_id_group']=2
    df.loc[(df.new_program_id_group=='Z_1')|(df.new_program_id_group=='Z_2')|(df.new_program_id_group=='Z_3')|(df.new_program_id_group=='T_2')|(df.new_program_id_group=='T_3')|(df.new_program_id_group=='T_4'),'new_program_id_group']=3
    df.loc[(df.new_program_id_group=='U_1'),'new_program_id_group']=4
    df.loc[(df.new_program_id_group=='V_1')|(df.new_program_id_group=='U_2'),'new_program_id_group']=5
    df.loc[(df.new_program_id_group=='V_3')|(df.new_program_id_group=='S_2')|(df.new_program_id_group=='V_4')|(df.new_program_id_group=='V_2'),'new_program_id_group']=6
    df.loc[(df.new_program_id_group=='T_1')|(df.new_program_id_group=='S_1'),'new_program_id_group']=7
    df.drop('program_id',axis=1,inplace=True)

    #creating col test_id and rating category together
    
    train=pd.read_csv('train_HK6lq50.csv')
    test=pd.read_csv('test_2nAIblo.csv')
    total=train.append(test)
    count=0
    total['test_id_and_rating']=0
    for a in total.trainee_engagement_rating.unique():
        for b in total.test_id.unique():
            count+=1
            total.loc[(total.trainee_engagement_rating==a)&(total.test_id==b),'test_id_and_rating']=count
    dic=dict(zip(total['id'],total['test_id_and_rating']))
    df['test_id_and_rating']=df['id'].map(dic)
    
    
    count=0
    total['test_id_and_education']=0
    for a in total.education.unique():
        for b in total.test_id.unique():
            count+=1
            total.loc[(total.education==a)&(total.test_id==b),'test_id_and_education']=count
    dic=dict(zip(total['id'],total['test_id_and_education']))
    df['test_id_and_education']=df['id'].map(dic)
    
    
    
    count=0
    total['program_type_and_rating']=0
    for a in total.trainee_engagement_rating.unique():
        for b in total.program_type.unique():
            count+=1
            total.loc[(total.trainee_engagement_rating==a)&(total.program_type==b),'program_type_and_rating']=count
    dic=dict(zip(total['id'],total['program_type_and_rating']))
    df['program_type_and_rating']=df['id'].map(dic)
    
    
    
    
    #grouping of test_id_and_rating

    c=pd.crosstab(df.test_id_and_rating,df.is_pass)
    c_pct=c.div(c.sum(1).astype(float),axis=0)
    c_pct.columns = ['fail', 'pass']
    c_pct['id_group']=pd.DataFrame(c_pct['pass'])

    c_pct.loc[(c_pct.id_group>=.20)&(c_pct.id_group<.30),'id_group']=1
    c_pct.loc[(c_pct.id_group>=.30)&(c_pct.id_group<.40),'id_group']=2
    c_pct.loc[(c_pct.id_group>=.40)&(c_pct.id_group<.50),'id_group']=3
    c_pct.loc[(c_pct.id_group>=.50)&(c_pct.id_group<.60),'id_group']=4
    c_pct.loc[(c_pct.id_group>=.60)&(c_pct.id_group<.70),'id_group']=5
    c_pct.loc[(c_pct.id_group>=.70)&(c_pct.id_group<.80),'id_group']=6
    c_pct.loc[(c_pct.id_group>=.80)&(c_pct.id_group<.90),'id_group']=7
    c_pct.loc[(c_pct.id_group>=.90)&(c_pct.id_group<1),'id_group']=8

    c_pct.id_group=c_pct.id_group.astype(int)
    c_pct.drop(['fail','pass'],axis=1,inplace=True)
    dic=c_pct.to_dict()
    dic4=dic['id_group']

    df['test_id_and_rating_group']=df['test_id_and_rating'].map(dic4).astype(int)
    
    #grouping of program_type_and_rating

    c=pd.crosstab(df.program_type_and_rating,df.is_pass)
    c_pct=c.div(c.sum(1).astype(float),axis=0)
    c_pct.columns = ['fail', 'pass']
    c_pct['id_group']=pd.DataFrame(c_pct['pass'])

    c_pct.loc[(c_pct.id_group>=.20)&(c_pct.id_group<.30),'id_group']=1
    c_pct.loc[(c_pct.id_group>=.30)&(c_pct.id_group<.40),'id_group']=2
    c_pct.loc[(c_pct.id_group>=.40)&(c_pct.id_group<.50),'id_group']=3
    c_pct.loc[(c_pct.id_group>=.50)&(c_pct.id_group<.60),'id_group']=4
    c_pct.loc[(c_pct.id_group>=.60)&(c_pct.id_group<.70),'id_group']=5
    c_pct.loc[(c_pct.id_group>=.70)&(c_pct.id_group<.80),'id_group']=6
    c_pct.loc[(c_pct.id_group>=.80)&(c_pct.id_group<.90),'id_group']=7
    c_pct.loc[(c_pct.id_group>=.90)&(c_pct.id_group<1),'id_group']=8

    c_pct.id_group=c_pct.id_group.astype(int)
    c_pct.drop(['fail','pass'],axis=1,inplace=True)
    dic=c_pct.to_dict()
    dic41=dic['id_group']

    df['program_type_and_rating_group']=df['program_type_and_rating'].map(dic41).astype(int)
    

    #col avg_rating by test_id
    
    total=train.append(test)
    c=pd.crosstab(total.test_id,total.trainee_engagement_rating)   #use this for final submission
    c['avg_rating']=(c[1.0]+2*c[2.0]+3*c[3.0]+4*c[4.0]+5*c[5.0])/(c[1.0]+c[2.0]+c[3.0]+c[4.0]+c[5.0])
    c['test_id']=c.index
    dic5=dict(zip(c['test_id'],c['avg_rating']))
    df['avg_rating']=df['test_id'].map(dic5)
    
    #rating_diff(count(1.0+2.0)-count(4.0+5.0))
    #c=pd.crosstab(total.test_id,total.trainee_engagement_rating)   #use this for final submission

    c=pd.crosstab(df.test_id,df.trainee_engagement_rating)
    c['rating_diff_test_id']=c[1.0]+c[2.0]-c[4.0]-c[5.0]+c[3.0]
    c['test_id']=c.index
    dic6=dict(zip(c['test_id'],c['rating_diff_test_id']))
    df['rating_diff_test_id']=df['test_id'].map(dic6)
    
    #col avg_rating by trainee_id
    #c=pd.crosstab(total.test_id,total.trainee_engagement_rating)   #use this for final submission

    c=pd.crosstab(df.trainee_id,df.trainee_engagement_rating)
    c['avg_rating_trainee_id']=(c[1.0]+2*c[2.0]+3*c[3.0]+4*c[4.0]+5*c[5.0])/(c[1.0]+c[2.0]+c[3.0]+c[4.0]+c[5.0])
    c['trainee_id']=c.index
    dic7=dict(zip(c['trainee_id'],c['avg_rating_trainee_id']))
    df['avg_rating_trainee_id']=df['trainee_id'].map(dic7)
    
    #is_pass_diff wrt trainee_engagement_rating
    c=pd.crosstab(df.trainee_engagement_rating,df.is_pass)
    c['trainee_engagement_rating']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_rating']=c['pass']-c['fail']
    dic8=dict(zip(c['trainee_engagement_rating'],c['is_pass_diff_rating']))
    df['is_pass_diff_rating']=df['trainee_engagement_rating'].map(dic8).astype(int)
    
    #is_pass_diff wrt total_programs_enrolled
    c=pd.crosstab(df.total_programs_enrolled,df.is_pass)
    c['total_programs_enrolled']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_total_programs_enrolled']=c['pass']-c['fail']
    dic9=dict(zip(c['total_programs_enrolled'],c['is_pass_diff_total_programs_enrolled']))
    df['is_pass_diff_total_programs_enrolled']=df['total_programs_enrolled'].map(dic9).astype(int)
    
    #is_pass_diff wrt difficulty_level_val
    c=pd.crosstab(df.difficulty_level_val,df.is_pass)
    c['difficulty_level_val']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_difficulty_level']=c['pass']-c['fail']
    dic10=dict(zip(c['difficulty_level_val'],c['is_pass_diff_difficulty_level']))
    df['is_pass_diff_difficulty_level']=df['difficulty_level_val'].map(dic10).astype(int)
    
    #is_pass_diff wrt education_val
    c=pd.crosstab(df.education_val,df.is_pass)
    c['education_val']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_education']=c['pass']-c['fail']
    dic11=dict(zip(c['education_val'],c['is_pass_diff_education']))
    df['is_pass_diff_education']=df['education_val'].map(dic11).astype(int)
    
    #is_pass_diff wrt city_tier
    c=pd.crosstab(df.city_tier,df.is_pass)
    c['city_tier']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_city_tier']=c['pass']-c['fail']
    dic12=dict(zip(c['city_tier'],c['is_pass_diff_city_tier']))
    df['is_pass_diff_city_tier']=df['city_tier'].map(dic12).astype(int)
    
    #is_pass_diff wrt new_program_id_group
    c=pd.crosstab(df.new_program_id_group,df.is_pass)
    c['new_program_id_group']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_new_program_id_group']=c['pass']-c['fail']
    dic13=dict(zip(c['new_program_id_group'],c['is_pass_diff_new_program_id_group']))
    df['is_pass_diff_new_program_id_group']=df['new_program_id_group'].map(dic13).astype(int)
    
    #is_pass_diff wrt program_id_val
    c=pd.crosstab(df.program_id_val,df.is_pass)
    c['program_id_val']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_program_id_val']=c['pass']-c['fail']
    dic14=dict(zip(c['program_id_val'],c['is_pass_diff_program_id_val']))
    df['is_pass_diff_program_id_val']=df['program_id_val'].map(dic14).astype(int)
    
    

    #is_pass_diff wrt program_duration
    c=pd.crosstab(df.program_duration,df.is_pass)
    c['program_duration']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_program_duration']=c['pass']-c['fail']
    dic15=dict(zip(c['program_duration'],c['is_pass_diff_program_duration']))
    df['is_pass_diff_program_duration']=df['program_duration'].map(dic15).astype(int)
    
    #is_pass_diff wrt total_test_taken
    c=pd.crosstab(df.total_test_taken,df.is_pass)
    c['total_test_taken']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_total_test_taken']=c['pass']-c['fail']
    dic16=dict(zip(c['total_test_taken'],c['is_pass_diff_total_test_taken']))
    df['is_pass_diff_total_test_taken']=df['total_test_taken'].map(dic16).astype(int)
    
    #is_pass_diff wrt test_type_val
    c=pd.crosstab(df.test_type_val,df.is_pass)
    c['test_type_val']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_test_type_val']=c['pass']-c['fail']
    dic17=dict(zip(c['test_type_val'],c['is_pass_diff_test_type_val']))
    df['is_pass_diff_test_type_val']=df['test_type_val'].map(dic17).astype(int)
    
    #is_pass_diff wrt program_type_val
    c=pd.crosstab(df.program_type_val,df.is_pass)
    c['program_type_val']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_program_type_val']=c['pass']-c['fail']
    dic18=dict(zip(c['program_type_val'],c['is_pass_diff_program_type_val']))
    df['is_pass_diff_program_type_val']=df['program_type_val'].map(dic18).astype(int)
    
    #is_pass_diff wrt total_trainee_in_each_test
    c=pd.crosstab(df.total_trainee_in_each_test,df.is_pass)
    c['total_trainee_in_each_test']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff_total_trainee_in_each_test']=c['pass']-c['fail']
    dic19=dict(zip(c['total_trainee_in_each_test'],c['is_pass_diff_total_trainee_in_each_test']))
    df['is_pass_diff_total_trainee_in_each_test']=df['total_trainee_in_each_test'].map(dic19).astype(int)
    
    


    #grouping for test_id
    c=pd.crosstab(df.test_id,df.is_pass)
    c_pct=c.div(c.sum(1).astype(float),axis=0)
    c_pct.columns = ['fail', 'pass']
    c_pct['id_group']=pd.DataFrame(c_pct['pass'])

    c_pct.loc[(c_pct.id_group>=.20)&(c_pct.id_group<.30),'id_group']=1
    c_pct.loc[(c_pct.id_group>=.30)&(c_pct.id_group<.40),'id_group']=2
    c_pct.loc[(c_pct.id_group>=.40)&(c_pct.id_group<.50),'id_group']=3
    c_pct.loc[(c_pct.id_group>=.50)&(c_pct.id_group<.60),'id_group']=4
    c_pct.loc[(c_pct.id_group>=.60)&(c_pct.id_group<.70),'id_group']=5
    c_pct.loc[(c_pct.id_group>=.70)&(c_pct.id_group<.80),'id_group']=6
    c_pct.loc[(c_pct.id_group>=.80)&(c_pct.id_group<.90),'id_group']=7
    c_pct.loc[(c_pct.id_group>=.90)&(c_pct.id_group<1),'id_group']=8

    c_pct.id_group=c_pct.id_group.astype(int)
    c_pct.drop(['fail','pass'],axis=1,inplace=True)
    dic=c_pct.to_dict()
    dic20=dic['id_group']

    df['test_id_group']=df['test_id'].map(dic20).astype(int)

    #grouping for trainee_id
    c=pd.crosstab(df.trainee_id,df.is_pass)
    c_pct=c.div(c.sum(1).astype(float),axis=0)
    c_pct.columns = ['fail', 'pass']
    c_pct['id_group']=pd.DataFrame(c_pct['pass'])
    c_pct.loc[(c_pct.id_group>=0)&(c_pct.id_group<.20),'id_group']=1
    c_pct.loc[(c_pct.id_group>=.20)&(c_pct.id_group<.40),'id_group']=2
    c_pct.loc[(c_pct.id_group>=.40)&(c_pct.id_group<.60),'id_group']=3
    c_pct.loc[(c_pct.id_group>=.60)&(c_pct.id_group<.80),'id_group']=4
    c_pct.loc[(c_pct.id_group>=.80)&(c_pct.id_group<=1),'id_group']=5

    c_pct.id_group=c_pct.id_group.astype(int)
    c_pct.drop(['fail','pass'],axis=1,inplace=True)
    dic=c_pct.to_dict()
    dic21=dic['id_group']

    df['trainee_id_group']=df['trainee_id'].map(dic21)
    
    #is_pass_diff wrt trainee_id
    c=pd.crosstab(df.trainee_id,df.is_pass)
    c['trainee_id']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff']=c['pass']-c['fail']
    dic22=dict(zip(c['trainee_id'],c['is_pass_diff']))
    df['is_pass_diff']=df['trainee_id'].map(dic22)
    

    #is_pass_diff2 wrt test_id
    c=pd.crosstab(df.test_id,df.is_pass)
    c['test_id']=c.index
    c['pass']=c[1]
    c['fail']=c[0]
    c['is_pass_diff2']=c['pass']-c['fail']
    dic23=dict(zip(c['test_id'],c['is_pass_diff2']))
    df['is_pass_diff2']=df['test_id'].map(dic23)
    
    col=['program_duration', 'city_tier', 'total_programs_enrolled',
       'trainee_engagement_rating', 'isage_null', 'test_type_val',
       'program_id_val', 'difficulty_level_val',
        'education_val', 'is_handicapped_val',
       'trainee_engagement_rating_mean_target','new_program_id_group']
    mean_enc=[]
    for i in col:
        means=df.groupby(i).is_pass.mean()
        df[i+'_mean_target']=df[i].map(means)
        df.drop(i,axis=1,inplace=True)
        mean_enc.append(means)

    df.drop('is_pass',axis=1,inplace=True)
    df.drop(['id','gender'],axis=1,inplace=True)
    
    
    dic_all=[dic1,dic2,dic3,dic4,dic41,dic5,dic6,dic7,dic8,dic9,dic10,dic11,dic12,dic13,dic14,dic15,dic16,dic17,dic18,dic19,dic20,dic21,dic22,dic23]
    
    return(df,dic_all,mean_enc)


def test_data_preprocess(df,train,test,dic_all,mean_enc):
    
    (dic1,dic2,dic3,dic4,dic41,dic5,dic6,dic7,dic8,dic9,dic10,dic11,dic12,dic13,dic14,dic15,dic16,dic17,dic18,dic19,dic20,dic21,dic22,dic23)=dic_all
    
    df['trainee_engagement_rating'].fillna(value=1.0,inplace=True)   

    df['isage_null']=0
    df.isage_null[df.age.isnull()]=1                                
    df['age'].fillna(value=0,inplace=True)        

    #new cols actual_programs_enrolled and total_test_taken
    df['actual_programs_enrolled']=df['trainee_id'].map(dic1).astype(int)
    df['total_test_taken']=df['trainee_id'].map(dic2).astype(int)


    #new col total_trainee_in_each_test
    df['total_trainee_in_each_test']=df['test_id'].map(dic3).astype(int)


    #LABEL ENCODING
    
    test_type=sorted(df['test_type'].unique())
    test_type_mapping=dict(zip(test_type,range(1,len(test_type)+1)))
    df['test_type_val']=df['test_type'].map(test_type_mapping).astype(int)
    df.drop('test_type',axis=1,inplace=True)

    program_type=sorted(df['program_type'].unique())
    program_type_mapping=dict(zip(program_type,range(1,len(program_type)+1)))
    df['program_type_val']=df['program_type'].map(program_type_mapping).astype(int)
    df.drop('program_type',axis=1,inplace=True)

    program_id=sorted(df['program_id'].unique())
    program_id_mapping=dict(zip(program_id,range(1,len(program_id)+1)))
    df['program_id_val']=df['program_id'].map(program_id_mapping).astype(int)
    #df.drop('program_id',axis=1,inplace=True)

    difficulty_level=['easy','intermediate','hard','vary hard']
    difficulty_level_mapping=dict(zip(difficulty_level,range(1,len(difficulty_level)+1)))
    df['difficulty_level_val']=df['difficulty_level'].map(difficulty_level_mapping).astype(int)
    df.drop('difficulty_level',axis=1,inplace=True)

    education=['No Qualification','High School Diploma','Matriculation','Bachelors','Masters']
    educationmapping=dict(zip(education,range(1,len(education)+1)))
    df['education_val']=df['education'].map(educationmapping).astype(int)
    df.drop('education',axis=1,inplace=True)

    is_handicapped=sorted(df['is_handicapped'].unique())
    is_handicappedmapping=dict(zip(is_handicapped,range(1,len(is_handicapped)+1)))
    df['is_handicapped_val']=df['is_handicapped'].map(is_handicappedmapping).astype(int)
    df.drop('is_handicapped',axis=1,inplace=True)

    #creating new program_id group based on is_pass percentage
    df['new_program_id_group']=pd.DataFrame(df['program_id'])
    df.loc[(df.new_program_id_group=='X_1')|(df.new_program_id_group=='X_3'),'new_program_id_group']=1
    df.loc[(df.new_program_id_group=='Y_1')|(df.new_program_id_group=='Y_2')|(df.new_program_id_group=='Y_3')|(df.new_program_id_group=='Y_4')|(df.new_program_id_group=='X_2'),'new_program_id_group']=2
    df.loc[(df.new_program_id_group=='Z_1')|(df.new_program_id_group=='Z_2')|(df.new_program_id_group=='Z_3')|(df.new_program_id_group=='T_2')|(df.new_program_id_group=='T_3')|(df.new_program_id_group=='T_4'),'new_program_id_group']=3
    df.loc[(df.new_program_id_group=='U_1'),'new_program_id_group']=4
    df.loc[(df.new_program_id_group=='V_1')|(df.new_program_id_group=='U_2'),'new_program_id_group']=5
    df.loc[(df.new_program_id_group=='V_3')|(df.new_program_id_group=='S_2')|(df.new_program_id_group=='V_4')|(df.new_program_id_group=='V_2'),'new_program_id_group']=6
    df.loc[(df.new_program_id_group=='T_1')|(df.new_program_id_group=='S_1'),'new_program_id_group']=7
    df.drop('program_id',axis=1,inplace=True)

    #creating col test_id and rating category
    
    total=train.append(test)
    count=0
    total['test_id_and_rating']=0
    for a in total.trainee_engagement_rating.unique():
        for b in total.test_id.unique():
            count+=1
            total.loc[(total.trainee_engagement_rating==a)&(total.test_id==b),'test_id_and_rating']=count
    dic=dict(zip(total['id'],total['test_id_and_rating']))
    df['test_id_and_rating']=df['id'].map(dic)
    
    
    count=0
    total['test_id_and_education']=0
    for a in total.education.unique():
        for b in total.test_id.unique():
            count+=1
            total.loc[(total.education==a)&(total.test_id==b),'test_id_and_education']=count
    dic=dict(zip(total['id'],total['test_id_and_education']))
    df['test_id_and_education']=df['id'].map(dic)
    
    count=0
    total['program_type_and_rating']=0
    for a in total.trainee_engagement_rating.unique():
        for b in total.program_type.unique():
            count+=1
            total.loc[(total.trainee_engagement_rating==a)&(total.program_type==b),'program_type_and_rating']=count
    dic=dict(zip(total['id'],total['program_type_and_rating']))
    df['program_type_and_rating']=df['id'].map(dic)
    
    
    
    
    
    
    
    
    
    
    
    
    #grouping of test_id_and_rating

    df['test_id_and_rating_group']=df['test_id_and_rating'].map(dic4)
    
    #grouping of program_type_and_rating

    df['program_type_and_rating_group']=df['program_type_and_rating'].map(dic41).astype(int)
    


    #col avg_rating by test_id
    
    df['avg_rating']=df['test_id'].map(dic5)
    
    #rating_diff(count(1.0+2.0)-count(4.0+5.0))
    #c=pd.crosstab(total.test_id,total.trainee_engagement_rating)   #use this for final submission

    df['rating_diff_test_id']=df['test_id'].map(dic6)
    
    #col avg_rating by trainee_id
    #c=pd.crosstab(total.test_id,total.trainee_engagement_rating)   #use this for final submission

    df['avg_rating_trainee_id']=df['trainee_id'].map(dic7)
    
    #is_pass_diff wrt trainee_engagement_rating
    df['is_pass_diff_rating']=df['trainee_engagement_rating'].map(dic8).astype(int)
    
    #is_pass_diff wrt total_programs_enrolled
    df['is_pass_diff_total_programs_enrolled']=df['total_programs_enrolled'].map(dic9).astype(int)
    
    #is_pass_diff wrt difficulty_level_val
    df['is_pass_diff_difficulty_level']=df['difficulty_level_val'].map(dic10).astype(int)
    
    #is_pass_diff wrt education_val
    df['is_pass_diff_education']=df['education_val'].map(dic11).astype(int)
    
    #is_pass_diff wrt city_tier
    df['is_pass_diff_city_tier']=df['city_tier'].map(dic12).astype(int)
    
    #is_pass_diff wrt new_program_id_group
    df['is_pass_diff_new_program_id_group']=df['new_program_id_group'].map(dic13).astype(int)
    
    #is_pass_diff wrt program_id_val
    df['is_pass_diff_program_id_val']=df['program_id_val'].map(dic14).astype(int)
    
    

    #is_pass_diff wrt program_duration
    df['is_pass_diff_program_duration']=df['program_duration'].map(dic15).astype(int)
    
    #is_pass_diff wrt total_test_taken
    df['is_pass_diff_total_test_taken']=df['total_test_taken'].map(dic16).astype(int)
    
    #is_pass_diff wrt test_type_val
    df['is_pass_diff_test_type_val']=df['test_type_val'].map(dic17).astype(int)
    
    #is_pass_diff wrt program_type_val
    df['is_pass_diff_program_type_val']=df['program_type_val'].map(dic18).astype(int)
    
    #is_pass_diff wrt total_trainee_in_each_test
    df['is_pass_diff_total_trainee_in_each_test']=df['total_trainee_in_each_test'].map(dic19).astype(int)
    
    


    #grouping for test_id
    df['test_id_group']=df['test_id'].map(dic20).astype(int)

    #grouping for trainee_id
    df['trainee_id_group']=df['trainee_id'].map(dic21)
    
    #is_pass_diff wrt trainee_id
    df['is_pass_diff']=df['trainee_id'].map(dic22)
    

    #is_pass_diff2 wrt test_id
    df['is_pass_diff2']=df['test_id'].map(dic23)
    
    
    #TARGET ENCODING
    col=['program_duration', 'city_tier', 'total_programs_enrolled',
       'trainee_engagement_rating', 'isage_null', 'test_type_val',
       'program_id_val', 'difficulty_level_val',
        'education_val', 'is_handicapped_val',
       'trainee_engagement_rating_mean_target','new_program_id_group']
    j=0
    for i in col:
        df[i+'_mean_target']=df[i].map(mean_enc[j])
        df.drop(i,axis=1,inplace=True)
        j+=1

    df.drop(['id','gender'],axis=1,inplace=True)
    
    
    return(df)


df=pd.read_csv('train_HK6lq50.csv')
train=pd.read_csv('train_HK6lq50.csv')
test=pd.read_csv('test_2nAIblo.csv')


df_train,dic_all,mean_enc=train_data_preprocess(df,train,test)

test=pd.read_csv('test_2nAIblo.csv')
df_test=test_data_preprocess(test,train,test,dic_all,mean_enc)

df_test.trainee_id_group.fillna(value=6.0,inplace=True)   
df_test.is_pass_diff.fillna(value=0.0,inplace=True)
df_test.test_id_and_rating_group.fillna(value=9.0,inplace=True)
df_test.avg_rating_trainee_id.fillna(value=3.0,inplace=True)


# ENSEMBLING RANDOMFORESTREGRESSOR WITH BAGGING 

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
y=train['is_pass']
x=df_train

model=RandomForestRegressor()
bags=10
seed=1
bagged_prediction=np.zeros(test.shape[0])
for n in range(0,bags):
    model.set_params(random_state=seed+n,n_estimators = 450, oob_score = True, n_jobs = -1,max_features = 2, min_samples_leaf = 10)
    
    model.fit(x,y)
    pred=model.predict(df_test)
    bagged_prediction+=pred
bagged_prediction/=bags

submission=pd.read_csv("sample_submission_vaSxamm.csv")
submission['is_pass']=bagged_prediction
submission.to_csv('RandomForestRegressor_bagging.csv', index=False)
