import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

label_to_description = {'LABEL_0': "The evidence supports the claim", 'LABEL_1': "The evidence refutes the claim", 'LABEL_2': "There isn't enough evidence provided", 'LABEL_3': "The model is uncertain"}

st.title('Climate Fact Checker')
st.image('https://www.itu.int/en/mediacentre/backgrounders/PublishingImages/climate-change-backgrounder.jpg')
st.subheader('Fact Check Your Climate Claims:')
tokenizer = AutoTokenizer.from_pretrained("spicytaco17/model")
model = AutoModelForSequenceClassification.from_pretrained("spicytaco17/model")

claim = st.text_input('Insert Claim: ')
evidence1 = st.text_input('Insert Evidence 1: ')
evidence2 = st.text_input('Insert Evidence 2: ')
click = st.button('Get Prediction')
if click:
  classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)
  classifier("Global warming is driving polar bears toward extinction||Rising global temperatures, caused by the greenhouse effect, contribute to habitat destruction, endangering various species, such as the polar bear.")
  prediction1 = classifier(claim+evidence1)
  prediction2 = classifier(claim+evidence2)

  label1 = prediction1[0]["label"]
  label2 = prediction2[0]["label"]

  confidence1 = prediction1[0]["score"]
  confidence2 = prediction2[0]["score"]

  st.balloons()

  st.text(f"The prediction for evidence 1 is: {label_to_description[label1]}")
  st.text(f"The prediction for evidence 2 is: {label_to_description[label2]}")

  st.text(f"The model predicted the first prediction with: {confidence1} certainty")
  st.text(f"The model predicted the second prediction with: {confidence2} certainty")

description = st.text('This model is based on the "Climate_Fever" dataset. If you would like to use the model on Hugging Face, check it out here: spicytaco17/model')
