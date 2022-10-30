import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

label_to_description = {'LABEL_0': "supports the claim", 'LABEL_1': "refutes the claim", 'LABEL_2': "doesn't provide enough context", 'LABEL_3': "is disputed"}

st.title('Climate Fact Checker')

description1 = st.text('This model is based on the "Climate_Fever" dataset. If you would like to use the')
description2 = st.text('model on Hugging Face, check it out here: spicytaco17/model')
st.image('https://www.itu.int/en/mediacentre/backgrounders/PublishingImages/climate-change-backgrounder.jpg')
st.subheader('Fact Check Your Climate Claims:')

tokenizer = AutoTokenizer.from_pretrained("spicytaco17/model")
model = AutoModelForSequenceClassification.from_pretrained("spicytaco17/model")

choice = st.selectbox('How many evidences are you using?', ['1 evidence', '2 evidences', '3 evidences', '4 evidences', '5 evidences'])
claim = st.text_input('Insert Claim: ')

if choice == '1 evidence':
  evidence1 = st.text_input('Insert Evidence 1: ')
if choice == '2 evidences':
  evidence1 = st.text_input('Insert Evidence 1: ')
  evidence2 = st.text_input('Insert Evidence 2: ')
if choice == '3 evidences':
  evidence1 = st.text_input('Insert Evidence 1: ')
  evidence2 = st.text_input('Insert Evidence 2: ')
  evidence3 = st.text_input('Insert Evidence 3: ')
if choice == '4 evidences':
  evidence1 = st.text_input('Insert Evidence 1: ')
  evidence2 = st.text_input('Insert Evidence 2: ')
  evidence3 = st.text_input('Insert Evidence 3: ')
  evidence4 = st.text_input('Insert Evidence 4: ')
if choice == '5 evidences':
  evidence1 = st.text_input('Insert Evidence 1: ')
  evidence2 = st.text_input('Insert Evidence 2: ')
  evidence3 = st.text_input('Insert Evidence 3: ')
  evidence4 = st.text_input('Insert Evidence 4: ')
  evidence5 = st.text_input('Insert Evidence 5: ')

click = st.button('Get Prediction')

def result():
  classifier = pipeline("text-classification", model = model, tokenizer = tokenizer)

  if choice == '1 evidence':
    prediction1 = classifier(claim+evidence1)
    label1 = prediction1[0]["label"]
    confidence1 = prediction1[0]["score"]
  if choice == '2 evidences':
    prediction1 = classifier(claim+evidence1)
    prediction2 = classifier(claim+evidence2)
    label1 = prediction1[0]["label"]
    label2 = prediction2[0]["label"]
    confidence1 = prediction1[0]["score"]
    confidence2 = prediction2[0]["score"]
  if choice == '3 evidences':
    prediction1 = classifier(claim+evidence1)
    prediction2 = classifier(claim+evidence2)
    prediction3 = classifier(claim+evidence3)
    label1 = prediction1[0]["label"]
    label2 = prediction2[0]["label"]
    label3 = prediction3[0]["label"]
    confidence1 = prediction1[0]["score"]
    confidence2 = prediction2[0]["score"]
    confidence3 = prediction3[0]["score"]
  if choice == '4 evidences':
    prediction1 = classifier(claim+evidence1)
    prediction2 = classifier(claim+evidence2)
    prediction3 = classifier(claim+evidence3)
    prediction4 = classifier(claim+evidence4)
    label1 = prediction1[0]["label"]
    label2 = prediction2[0]["label"]
    label3 = prediction3[0]["label"]
    label4 = prediction4[0]["label"]
    confidence1 = prediction1[0]["score"]
    confidence2 = prediction2[0]["score"]
    confidence3 = prediction3[0]["score"]
    confidence4 = prediction4[0]["score"]
  if choice == '5 evidences':
    prediction1 = classifier(claim+evidence1)
    prediction2 = classifier(claim+evidence2)
    prediction3 = classifier(claim+evidence3)
    prediction4 = classifier(claim+evidence4)
    prediction5 = classifier(claim+evidence5)
    label1 = prediction1[0]["label"]
    label2 = prediction2[0]["label"]
    label3 = prediction3[0]["label"]
    label4 = prediction4[0]["label"]
    label5 = prediction5[0]["label"]
    confidence1 = prediction1[0]["score"]
    confidence2 = prediction2[0]["score"]
    confidence3 = prediction3[0]["score"]
    confidence4 = prediction4[0]["score"]
    confidence5 = prediction5[0]["score"]

  st.subheader('Results:')
  st.balloons()
  st.text("-----------------------------------------------------------------------------------")
  col1, col2 = st.columns(2)

  if choice == '1 evidence':
    col1.write(f"Evidence 1: {label_to_description[label1]}")
    col2.write(f"The model predicted the first prediction with: {confidence1} certainty")
    st.text("-----------------------------------------------------------------------------------")
  if choice == '2 evidences':
    col1.write(f"Evidence 1: {label_to_description[label1]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 2: {label_to_description[label2]}")
    col2.write(f"The model predicted the first prediction with: {confidence1} certainty")
    col2.write(f"The model predicted the second prediction with: {confidence2} certainty")
    st.text("-----------------------------------------------------------------------------------")
  if choice == '3 evidences':
    col1.write(f"Evidence 1: {label_to_description[label1]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 2: {label_to_description[label2]}")
    col1.write("")
    col1.write(f"Evidence 3: {label_to_description[label3]}")
    col2.write(f"The model predicted the first prediction with: {confidence1} certainty")
    col2.write(f"The model predicted the second prediction with: {confidence2} certainty")
    col2.write(f"The model predicted the third prediction with: {confidence3} certainty")
    st.text("-----------------------------------------------------------------------------------")
  if choice == '4 evidences':
    col1.write(f"Evidence 1: {label_to_description[label1]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 2: {label_to_description[label2]}")
    col1.write("")
    col1.write(f"Evidence 3: {label_to_description[label3]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 4: {label_to_description[label4]}")
    col2.write(f"The model predicted the first prediction with: {confidence1} certainty")
    col2.write(f"The model predicted the second prediction with: {confidence2} certainty")
    col2.write(f"The model predicted the third prediction with: {confidence3} certainty")
    col2.write(f"The model predicted the fourth prediction with: {confidence4} certainty")
    st.text("-----------------------------------------------------------------------------------")
  if choice == '5 evidences':
    col1.write(f"Evidence 1: {label_to_description[label1]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 2: {label_to_description[label2]}")
    col1.write("")
    col1.write(f"Evidence 3: {label_to_description[label3]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 4: {label_to_description[label4]}")
    col1.write("")
    col1.write("")
    col1.write(f"Evidence 5: {label_to_description[label5]}")
    col2.write(f"The model predicted the first prediction with: {confidence1} certainty")
    col2.write(f"The model predicted the second prediction with: {confidence2} certainty")
    col2.write(f"The model predicted the third prediction with: {confidence3} certainty")
    col2.write(f"The model predicted the fourth prediction with: {confidence4} certainty")
    col2.write(f"The model predicted the fifth prediction with: {confidence5} certainty")
    st.text("-----------------------------------------------------------------------------------")

if (choice == '1 evidence') and (click) and (claim != "") and (evidence1 != ""):
  result()

elif (choice == '2 evidences') and (click) and (claim != "") and ((evidence1 != "") and (evidence2 != "")):
  result()

elif (choice == '3 evidences') and (click) and (claim != "") and ((evidence1 != "") and (evidence2 != "") and (evidence3 != "")):
  result()

elif (choice == '4 evidences') and (click) and (claim != "") and ((evidence1 != "") and (evidence2 != "") and (evidence3 != "") and (evidence4 != "")):
  result()

elif (choice == '5 evidences') and (click) and (claim != "") and ((evidence1 != "") and (evidence2 != "") and (evidence3 != "") and (evidence4 != "") and (evidence5 != "")):
  result()

else:
  st.error("Invalid input: all sections must be filled in")
