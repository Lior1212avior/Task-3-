import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score  # Import r2_score
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle


 
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    

# תיקון שמות המודל 
    results = []
    for index, row in df.iterrows():
        result = re.sub(re.escape(row['manufactor']), '', row['model'])
        results.append(result)

# Add the results to a new column in the DataFrame
    df['model'] = results

    
     # יצירת תכונה חדשה נוספת במודל 
    
    for index, row in df.iterrows():
        if (row['Pic_num'] == 0) or (row['Pic_num'] == ''):
            df.at[index, 'Pic_Score'] = 0
        elif row['Pic_num'] == 1:
            df.at[index, 'Pic_Score'] = 1
        elif row['Pic_num'] >= 2 and row['Pic_num'] < 5:
            df.at[index, 'Pic_Score'] = 2
        elif row['Pic_num'] >= 5 and row['Pic_num'] < 8:
            df.at[index, 'Pic_Score'] = 3
        elif row['Pic_num'] >= 8 and row['Pic_num'] < 15:
            df.at[index, 'Pic_Score'] = 4
    
    
    

    # מילוי ערכים חסרים
    df.fillna({
        'Hand': df['Hand'].mode()[0],
        'Gear': df['Gear'].mode()[0],
        'capacity_Engine': df['capacity_Engine'].mode()[0],
        'Prev_ownership': df['Prev_ownership'].mode()[0],
        'Curr_ownership': df['Curr_ownership'].mode()[0],
        'Area': df['Area'].mode()[0],
        'City': df['City'].mode()[0],
        'Pic_num': df['Pic_num'].mode()[0],
        'Description': '',
        'Color': df['Color'].mode()[0],
        'Km': df['Km'].mode()[0],
        'Test': df['Test'].mode()[0],
        'Pic_Score': df['Pic_Score'].mode()[0],
        'Supply_score': df['Supply_score'].mode()[0]
        
    }, inplace=True)
   
    
    # טיפול בעמודת התאריך
    df['Cre_date'] = pd.to_datetime(df['Cre_date'], format='%d/%m/%Y', errors='coerce')
    df['Repub_date'] = pd.to_datetime(df['Repub_date'], format='%d/%m/%Y', errors='coerce')


    # Alternatively, convert to a more interpretable numeric representation (e.g., day of the year)
    df['Cre_date'] = df['Cre_date'].dt.dayofyear  # Day of the year as int
    df['Repub_date'] = df['Repub_date'].dt.dayofyear  # Day of the year as int
    
   
    # STR-הפיכת המשתנים ל
    categorical_columns = ['manufactor', 'model', 'Gear', 'Engine_type', 'Area', 'City', 'Color', 'Prev_ownership', 'Curr_ownership']
    for col in categorical_columns:
        df[col] = df[col].astype(str)
   
    # קידוד משתנים קטגוריאליים
    le = LabelEncoder()
    df['manufactor'] = le.fit_transform(df['manufactor'])
    df['model'] = le.fit_transform(df['model'])
    df['Gear'] = le.fit_transform(df['Gear'])
    df['Engine_type'] = le.fit_transform(df['Engine_type'])
    df['Area'] = le.fit_transform(df['Area'])
    df['City'] = le.fit_transform(df['City'])
    df['Color'] = le.fit_transform(df['Color'])
    df['Prev_ownership'] = le.fit_transform(df['Prev_ownership'])
    df['Curr_ownership'] = le.fit_transform(df['Curr_ownership'])
    df['Pic_Score'] = le.fit_transform(df['Pic_Score'])
   
    # הסרת פסיקים והמרת העמודות הערכיות לפורמט float
    
    df['Km'] = df['Km'].str.replace(',', '').replace('None', '0').astype(float)
    #df['Km'] = df['Km'].replace('None', '0').astype(float)
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '').astype(float)
    df['Year'].astype(int)
    
    # יצירת תכונות חדשות
    
    df['Car_age'] = 2024 - df['Year']
    df['Km_Per_year'] = df['Km'] - df['Car_age']

   
    # נרמול התכונות
    scaler = StandardScaler()
    numeric_features = ['Supply_score','Prev_ownership', 'Curr_ownership']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    
    df.drop(columns=['Description','Cre_date','Repub_date'],inplace=True)
   
    return df



