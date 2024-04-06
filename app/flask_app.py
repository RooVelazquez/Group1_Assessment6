from flask import Flask, request, send_from_directory, render_template
import pandas as pd
import os
import pickle
import numpy as np
import json
from joblib import load

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(app.root_path, 'results')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load your model with an absolute path or relative to app.root_path
model_path = os.path.join(app.root_path, '../models/grid_search_decision_tree.sav')
model = pickle.load(open(model_path, 'rb'))


@app.route('/')
def index():
    return render_template('upload.html')  # Assuming you have an upload.html in your templates folder


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Save uploaded file
        f = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)

        # Process and predict
        result_filepath = process_and_predict(filepath)

        # Return link to download the predictions
        return f'''
        Prediction complete. <a href="/results/{os.path.basename(result_filepath)}">Download Results</a>
        '''


def fill_nulls(data, imputation_dict):
    pack_list = ['K01', '102', '105', 'O01', '103', '101', '107', '301', '104', '108', '109', 'M01']
    pack_list.remove('108')
    pack_list.remove('M01')

    for item in data['PACK'].unique():
        data.loc[data['PACK'] == item, 'AMOUNT_RUB_CLO_PRC'] = imputation_dict['AMOUNT_RUB_CLO_PRC'][item]
        data.loc[data['PACK'] == item, 'AMOUNT_RUB_NAS_PRC'] = imputation_dict['AMOUNT_RUB_NAS_PRC'][item]
        data.loc[data['PACK'] == item, 'TRANS_COUNT_NAS_PRC'] = imputation_dict['TRANS_COUNT_NAS_PRC'][item]

    for item in data['PACK'].unique():
        data.loc[data['PACK'] == item, 'AMOUNT_RUB_SUP_PRC'] = imputation_dict['AMOUNT_RUB_SUP_PRC'][item]
        data.loc[data['PACK'] == item, 'TRANS_COUNT_SUP_PRC'] = imputation_dict['TRANS_COUNT_SUP_PRC'][item]
        data.loc[data['PACK'] == item, 'TRANS_COUNT_ATM_PRC'] = imputation_dict['TRANS_COUNT_ATM_PRC'][item]
        data.loc[data['PACK'] == item, 'AMOUNT_RUB_ATM_PRC'] = imputation_dict['AMOUNT_RUB_ATM_PRC'][item]
        data.loc[data['PACK'] == item, 'CNT_TRAN_ATM_TENDENCY1M'] = imputation_dict['CNT_TRAN_ATM_TENDENCY1M'][item]
        data.loc[data['PACK'] == item, 'SUM_TRAN_ATM_TENDENCY1M'] = imputation_dict['SUM_TRAN_ATM_TENDENCY1M'][item]

    for item in data['PACK'].unique():
        data.loc[data['PACK'] == item, 'CNT_TRAN_SUP_TENDENCY3M'] = imputation_dict['CNT_TRAN_SUP_TENDENCY3M'][item]
        data.loc[data['PACK'] == item, 'SUM_TRAN_SUP_TENDENCY3M'] = imputation_dict['SUM_TRAN_SUP_TENDENCY3M'][item]
        data.loc[data['PACK'] == item, 'CNT_TRAN_ATM_TENDENCY3M'] = imputation_dict['CNT_TRAN_ATM_TENDENCY3M'][item]
        data.loc[data['PACK'] == item, 'AMOUNT_RUB_ATM_PRC'] = imputation_dict['AMOUNT_RUB_ATM_PRC'][item]
        data.loc[data['PACK'] == item, 'SUM_TRAN_ATM_TENDENCY3M'] = imputation_dict['SUM_TRAN_ATM_TENDENCY3M'][item]
        data.loc[data['PACK'] == item, 'TRANS_AMOUNT_TENDENCY3M'] = imputation_dict['TRANS_AMOUNT_TENDENCY3M'][item]
        data.loc[data['PACK'] == item, 'TRANS_CNT_TENDENCY3M'] = imputation_dict['TRANS_CNT_TENDENCY3M'][item]

    return data


def one_hot_encode(data):
    oneh = load('../models/onehot_model.joblib')
    col_names = oneh.get_feature_names_out()
    object_cols = data.select_dtypes(include=['object']).columns
    data_dummies = pd.DataFrame(oneh.transform(data[object_cols]), columns=col_names)
    data_dummies.index = data.index
    data = pd.merge(data, data_dummies, left_index=True, right_index=True)
    data.drop(columns=object_cols, inplace=True)
    return data


def scaling(data):
    scaler = load('../models/scaling_model.joblib')
    data_scaled = scaler.transform(data)
    data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
    return data


def load_columns():
    with open('../models/columns.txt', 'r') as f:
        cols = [line.strip() for line in f]
    return list(cols)


def load_imput_dict():
    with open('../models/imput_dict.json', 'r') as f:
        # Load the data into a dictionary
        data_dict = json.load(f)
    return data_dict


def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def predict_data(data, cols, loaded_model):
    data['Prediction'] = loaded_model.predict(data[cols])
    return data


def pipeline(data, filename):
    print("Step 1:Fill Nulls\n\n")
    data = fill_nulls(data, load_imput_dict())
    print("Step 2:One Hot\n\n")
    data = one_hot_encode(data)
    print("Step 3:Scale\n\n")
    data = scaling(data)
    print("Step 4:Load Columns\n\n")
    cols = load_columns()
    print("Step 5:Load Model\n\n")
    loaded_model = load_model(filename)
    print("Step 6:Predict\n\n")
    data = predict_data(data, cols, loaded_model)

    data['ID'] = data.index
    data['Probabilities'] = loaded_model.predict_proba(data[cols])[:, 1]
    data = data[['ID', 'Prediction', 'Probabilities']]
    return data


def process_and_predict(filepath):
    data = pd.read_csv(filepath)
    predictions = pipeline(data, '../models/grid_search_decision_tree.sav')
    results_filename = os.path.basename(filepath).replace('.csv', '_results.csv')
    result_filepath = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
    pd.DataFrame(predictions).to_csv(result_filepath, index=False)
    return results_filename


@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run()
