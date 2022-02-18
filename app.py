from functions import *

# DATA
user_list = get_user_list()

feature_dict = get_feature_dict()
feature_to_ids = {v['name']:k for k,v in feature_dict.items()}
feature_list = list(feature_to_ids)
feature_list.remove('SK_ID_CURR')
feature_list.remove('TARGET')

# SIDEBAR
with st.sidebar:
    st.image('https://cdn.iconscout.com/icon/free/png-256/user-1648810-1401302.png', width=100)
    st.title('User selection')

    selected_user = str(st.sidebar.selectbox('ID', user_list))
    user_infos = get_user_infos(selected_user)

    st.write('Informations')
    st.table(user_infos)

    st.markdown('---')
    st.markdown('[![Github - App](https://img.shields.io/badge/Github-App-FF0000?logo=github)](https://github.com/leoguillaume/home_credit_default_risk_api) [![Github - API](https://img.shields.io/badge/Github-API-1E90FF?logo=github)](https://github.com/leoguillaume/home_credit_default_risk_api) [![Github - Model](https://img.shields.io/badge/Github-Model-32CD32?logo=github)](https://github.com/leoguillaume/home_credit_default_risk_api)', unsafe_allow_html=True)

# MAIN

st.title('Scoring Credit Dashboard')
st.markdown('---')

st.header('🔮 Credit risk prediction')

negative_proba, positive_proba = get_prediction(selected_user)

col1, col2 = st.columns(2)

with col1:
    st.success(f'Solvant probability: {negative_proba:.0%}')

with col2:
    st.error(f'Insolvant probability : {positive_proba:.0%}')

if st.button('View feature importances with shap'):
    explained_values, expected_value, user_data = get_shap_values(selected_user)
    st_shap(shap.force_plot(expected_value, explained_values, user_data))

st.markdown('---')

st.header('🔍 User analysis')

col3, col4 = st.columns(2)

with col3:
    selected_feature = st.selectbox('Feature', feature_list)
    selected_feature = str(feature_to_ids[selected_feature])
    st.info(feature_dict[selected_feature]['description'])

negative_dist, positive_dist = get_feature_data(selected_feature)
user_data = get_user_data(selected_user)
user_feature_data = user_data[feature_dict[selected_feature]['name']]


feature_data = pd.Series(np.concatenate([negative_dist, positive_dist]), name = feature_dict[selected_feature]['name'])
delta, delta_color = get_delta(user_feature_data, selected_feature, feature_dict, feature_data)
feature_name = feature_dict[selected_feature]['name'].capitalize().replace('_', ' ')

with col4:
    st.metric(f'{feature_name} of user {selected_user}', user_feature_data, delta, delta_color)

if feature_dict[selected_feature]['type'] == 'object':
    fig = get_categorical_chart(negative_dist, positive_dist, feature_name)
else:
    fig = get_numerical_chart(negative_dist, positive_dist, user_feature_data, feature_name.lower())

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')


