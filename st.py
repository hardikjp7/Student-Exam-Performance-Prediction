import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.title('Student Performance Prediction')

# Get user inputs
gender = st.selectbox('Gender', ['male', 'female'])
ethnicity = st.selectbox('Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
parental_level_of_education = st.selectbox('Parental Level of Education', ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox('Lunch', ['standard', 'free/reduced'])
test_preparation_course = st.selectbox('Test Preparation Course', ['none', 'completed'])
reading_score = st.slider('Reading Score', min_value=0, max_value=100, step=1)
writing_score = st.slider('Writing Score', min_value=0, max_value=100, step=1)

# Create data frame
data = CustomData(
    gender=gender,
    race_ethnicity=ethnicity,
    parental_level_of_education=parental_level_of_education,
    lunch=lunch,
    test_preparation_course=test_preparation_course,
    reading_score=reading_score,
    writing_score=writing_score
)
pred_df = data.get_data_as_data_frame()

# Display results
if st.button('Predict'):
    # Make prediction
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # Display prediction result
    # if isinstance(results, list):
    #     st.markdown(f"**Prediction Results:** {results[0]}")
    # else:
    #     st.markdown(f"**Prediction Results:** {results}")
    #st.write('Prediction Results:')
    #st.write(results[0])
    st.markdown(f"<h3>Prediction Results: {results[0]}</h3>", unsafe_allow_html=True)
