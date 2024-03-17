import streamlit as st
import requests

# Streamlit UI components
st.title("Text Question Answering System")

# Upload text file
text_file = st.file_uploader("Upload Text file", type=["txt"])

# Input question
question = st.text_input("Enter your question")

# Button to trigger QA
if st.button("Get Answer"):
    if text_file is not None and question.strip() != "":
        # Prepare data for the request
        data = {"question": question}
        files = {"text_file": text_file.read()}
        response = requests.post("http://localhost:8000/qa", data=data, files=files)
        # Send data to backend API endpoint
        #response = requests.post("http://localhost:8000/qa", json=data)

        # Display answer
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.write("Error occurred. Please try again.")
