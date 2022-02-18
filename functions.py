import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests, os
import plotly.graph_objects as go
import plotly.express as px
import shap
from math import log, floor

#API_ROOT = 'http://127.0.0.1:8000'
API_ROOT = 'https://leoguillaume-credit-scoring.herokuapp.com'

@st.cache
def get_user_list():

    json_response = requests.get(os.path.join(API_ROOT, 'user', 'user_list')).json()
    user_list = json_response['user_id_list']

    return user_list

#@st.cache
def get_prediction(user_id, amount, annuity):

    request = {'amount': amount, 'annuity': annuity}
    json_response = requests.post(os.path.join(API_ROOT, 'model', 'predict', user_id), json=request).json()
    negative_proba = json_response['negative_pred']
    positive_proba = json_response['positive_pred']

    return negative_proba, positive_proba 

@st.cache
def get_feature_dict():

    json_response = requests.get(os.path.join(API_ROOT, 'data', 'feature_list')).json()
    feature_dict = json_response['feature_list']

    return feature_dict

@st.cache
def get_feature_data(feature_id):

        json_response = requests.get(os.path.join(API_ROOT, 'data', 'feature_data', feature_id)).json()
        negative_dist = [v if v != 'null' else np.NaN for v in json_response['negative_data']]
        positive_dist = [v if v != 'null' else np.NaN for v in json_response['positive_data']]

        return negative_dist, positive_dist

@st.cache
def get_user_data(user_id):

    json_response = requests.get(os.path.join(API_ROOT, 'data', 'user_data', user_id)).json()
    user_data = pd.Series(json_response['user_data']).replace('null', 'N/A')

    return user_data

def price_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.0f%s' % (number / k**magnitude, units[magnitude])

def get_delta(user_feature_data, feature_id, feature_dict, feature_data):

    if user_feature_data == 'N/A':
        delta = None
        delta_color = 'off'
    else:
        if feature_dict[feature_id]['type'] == 'object':
            f = feature_data.value_counts(normalize=True)
            delta = f'Most common value : {f.index[0]} ({f[0]:.0%})'
            delta_color = 'off'
        else:
            m = feature_data.astype(feature_dict[feature_id]['type']).mean()
            u = float(user_feature_data)
            delta = f'{(u - m) / m:.0%}'
            delta_color = 'normal'

    return delta, delta_color
        
@st.cache
def get_user_infos(user_id):

    user_data = get_user_data(user_id)
    user_infos = user_data[['CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]
    user_infos['DAYS_BIRTH'] = user_data['DAYS_BIRTH'] if user_data['DAYS_BIRTH'] == 'N/A' else str(round(np.abs(int(user_data['DAYS_BIRTH']) / 365)))
    user_infos['AMT_INCOME_TOTAL'] = user_data['AMT_INCOME_TOTAL'] if user_data['AMT_INCOME_TOTAL'] == 'N/A' else price_format(float(user_data['AMT_INCOME_TOTAL']))
    user_infos['AMT_CREDIT'] = user_data['AMT_CREDIT'] if user_data['AMT_CREDIT'] == 'N/A' else price_format(float(user_data['AMT_CREDIT']))
    user_infos['AMT_ANNUITY'] = user_data['AMT_ANNUITY'] if user_data['AMT_ANNUITY'] == 'N/A' else price_format(float(user_data['AMT_ANNUITY']))
    user_infos.index = ['Gender', 'Age', 'Family status', 'Number of children', 'Income total', 'Credit amount', 'Credit annuities']
    user_infos.name = ''

    return user_infos

def get_numerical_chart(negative_dist, positive_dist, user_feature_data, feature_name):

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=negative_dist, name='Yes', marker_color='green'))
    fig.add_trace(go.Histogram(x=positive_dist, name='No', marker_color='red'))
    fig.update_layout(
        barmode='overlay',
         autosize=False, 
         title=f'Distribution of {feature_name}', 
         legend_title="Solvant")
    fig.update_traces(opacity=0.4)

    if user_feature_data != 'N/A':
        fig.add_vline(
            x=float(user_feature_data), 
            annotation_text='user', 
            annotation_position='top', 
            line_color='white',  
            line_width=3,
            line_dash="dash")
        
        return fig

def get_categorical_chart(negative_dist, positive_dist, feature_name):

    target = pd.Series(np.concatenate([np.zeros(len(negative_dist)), np.ones(len(positive_dist))]), name='Solvant')
    feature_data = pd.Series(np.concatenate([negative_dist, positive_dist]), name = feature_name)
    feature_data.replace('null', np.NaN, inplace=True)
    chart_data = pd.concat([feature_data, target], axis=1)
    chart_data.Solvant.replace({0.0:'Yes', 1.0:'No'}, inplace=True)

    fig = px.histogram(chart_data, x=feature_name, color="Solvant",  barmode='group', color_discrete_sequence=['green', 'red'], opacity=0.4)
    fig.update_yaxes(title='')
        
    return fig

@st.cache
def get_shap_values(user_id, amount, annuity):
    
    request = {'amount': amount, 'annuity': annuity}
    json_response = requests.post(os.path.join(API_ROOT, 'model', 'shap_values', user_id), json=request).json()
    explained_values = np.array(json_response['explained_values']).reshape(1, -1)
    expected_value = json_response['expected_value']
    user_data = pd.Series(json_response['user_data']).replace('null', np.NaN)

    return explained_values, expected_value, user_data

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

