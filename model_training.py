#import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score  # Import r2_score
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from car_data_prep import prepare_data
import joblib

input_data = 'C:\\Users\\USER\\flaskWebProject\\dataset.csv'
processed_data= prepare_data(input_data)

#utf8_file_path = 'C:\\Users\\USER\\flaskWebProject\\dataset_Correct.csv'
# Ensure df is not None
#if processed_data is not None:
    # Save the processed data to a new CSV file
#    processed_data.to_csv(utf8_file_path, index=False, encoding='utf-8-sig')
#    print('Completed')
#else:
#   print('Error: Data processing failed')



# Print feature names
print("Feature names in the processed DataFrame:")
print(processed_data.columns.tolist())

 
# Specify the path where you want to save the CSV file
#output_file_path = 'output_dataset.csv'

# Save the DataFrame to a CSV file
#processed_data.to_csv(output_file_path, index=False) 


# Define features and target
X = processed_data[['manufactor','Year', 'model','Hand','Gear', 'capacity_Engine','Engine_type','Km']]
#X = processed_data[['manufactor','Year','model','Gear', 'Engine_type','Area', 'City', 'Km', 'capacity_Engine','Supply_score','Car_age','Km_Per_year','Pic_Score']]

y = processed_data['Price']


# חלוקת הנתונים לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# בניית המודל
model = ElasticNet(alpha=0.5, l1_ratio=0.2 , random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)



# אימון המודל על כל סט האימון
model.fit(X_train, y_train)


# חיזוי על סט הבדיקה
y_pred = model.predict(X_test)



# Save the model to a pickle file
with open('trained_model.pkl', 'wb') as f:
   pickle.dump(model, f)


# Optionally, save the feature names for validation during prediction
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

 
# Save trained model
joblib.dump(model, 'trained_model.pkl')

print('came to end - model training')

    
