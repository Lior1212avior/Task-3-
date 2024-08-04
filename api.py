# api.py
from flask import Flask, request, render_template
import pandas as pd
import joblib
from car_data_prep import prepare_data
import csv

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')



# Function to load options from Excel sheet
def load_options_from_excel(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    options = df.to_dict(orient='records')
    return options


@app.route('/')
#def home():
#    return render_template('index.html')

def home():
    # Load manufactor options from Excel sheet
    manufactor_options = load_options_from_excel('options_categorial.xlsx', 'manufactor')
    # Load model options from Excel sheet
    model_options = load_options_from_excel('options_categorial.xlsx', 'model')
    # Load Gear options from Excel sheet
    Gear_options = load_options_from_excel('options_categorial.xlsx', 'Gear')
    # Load Engine_type options from Excel sheet
    Engine_type_options = load_options_from_excel('options_categorial.xlsx', 'Engine_type')

    # Load Area options from Excel sheet
    Area_options = load_options_from_excel('options_categorial.xlsx', 'Area')

    # Load City options from Excel sheet
    City_options = load_options_from_excel('options_categorial.xlsx', 'City')

    # Load Color options from Excel sheet
    Color_options = load_options_from_excel('options_categorial.xlsx', 'Color')

    # Load Prev_ownership options from Excel sheet
    Prev_ownership_options = load_options_from_excel('options_categorial.xlsx', 'Prev_owner')

   # Load Prev_ownership options from Excel sheet
    Curr_ownership_options = load_options_from_excel('options_categorial.xlsx', 'Curr_owner')
    
    return render_template('index.html', manufactor_options=manufactor_options, model_options=model_options, Gear_options=Gear_options ,Engine_type_options=Engine_type_options,  Area_options=Area_options , City_options=City_options , Prev_owner_options=Prev_ownership_options, Curr_owner_options=Curr_ownership_options,Color_options=Color_options)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    form = request.form

    # Get the actual value of manufactor and model
    manufactor_value = form['manufactor']
    model_value = form['model']
    Gear_value = form['Gear']
    Engine_type_value = form['Engine_type']
    Prev_owner_value = form['Prev_ownership']
    Curr_owner_value = form['Curr_ownership']
  
      
    car_data = {
        #'manufactor': str(form['manufactor']) if 'manufactor' in form else '',
        'manufactor': manufactor_value,
        'Year': int(form['Year']) if 'Year' in form and form['Year'] else 0,
        #'model': form['model'] if 'model' in form else '',
        'model': model_value,
        'Hand': int(form['Hand']) if 'Hand' in form and form['Hand'] else 0,
        #'Gear': form['Gear'] if 'Gear' in form else '',
        'Gear': Gear_value,
        'capacity_Engine': form['capacity_Engine'] if 'capacity_Engine' in form else '',
        #'Engine_type': form['Engine_type'] if 'Engine_type' in form else '',
        'Engine_type': Engine_type_value,
        #'Prev_ownership': form['Prev_ownership'] if 'Prev_ownership' in form else '',
        'Prev_ownership':Prev_owner_value,
        #'Curr_ownership': form['Curr_ownership'] if 'Curr_ownership' in form else '',
        'Curr_ownership':Curr_owner_value,
        'Area': form['Area'] if 'Area' in form else '',
        'City': form['City'] if 'City' in form else '',
        'Pic_num': form['Pic_num'] if 'Pic_num' in form else '',
        #'Cre_date': form['Cre_date'] if 'Cre_date' in form else '',
        #'Repub_date': form['Repub_date'] if 'Repub_date' in form else '',
        #'Description': form['Description'] if 'Description' in form else '',
        'Color': form['Color'] if 'Color' in form else '',
        'Km': float(form['Km']) if 'Km' in form and form['Km'] else 0.0,
        'Test': form['Test'] if 'Test' in form else '',
        'Supply_score': float(form['Supply_score']) if 'Supply_score' in form and form['Supply_score'] else 0.0
        
    }

    # Convert to DataFrame
    dfInput = pd.DataFrame([car_data])


# Specify the path where you want to save the CSV file
 #   output_file_path = 'User_dataset.csv'

# Save the DataFrame to a CSV file
 #   dfInput.to_csv(output_file_path, index=False, encoding = 'utf-8-sig')
    
      

    dfInput.drop(columns=['Prev_ownership','Curr_ownership', 'Area', 'City', 'Pic_num', 'Color', 'Test','Supply_score'],inplace=True)
    # Print feature names
    print("Feature names in the processed DataFrame:")
    print(dfInput.columns.tolist())    
    
    # Predict price
    prediction = model.predict(dfInput)

    # Return result
    return f'The predicted price is: {prediction[0]:.2f} ש"ח'

if __name__ == '__main__':
    app.run(debug=True)

