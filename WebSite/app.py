# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Resume Category Prediction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import time

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-section {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .prediction-section {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .file-uploader {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 1rem;
    }
    .category-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    # Header with animation
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #2c3e50; font-size: 2.5rem; margin-bottom: 1rem;'>
                Resume Category Prediction
            </h1>
            <p style='color: #7f8c EEEEd; font-size: 1.2rem;'>
                Upload your resume and discover its job category
            </p>
        </div>
    """, unsafe_allow_html=True)

    # File upload section with custom styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Your Resume")
    st.markdown("Supported formats: PDF, DOCX, TXT")
    
    uploaded_file = st.file_uploader("", type=["pdf", "docx", "txt"], key="file_uploader")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Simulate processing with progress bar
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing... {i+1}%")
                time.sleep(0.01)  # Small delay for visual effect
            
            # Extract text from the uploaded file
            resume_text = handle_file_upload(uploaded_file)
            
            # Prediction section with custom styling
            st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
            st.markdown("### 🎯 Prediction Results")
            
            # Make prediction with animation
            category = pred(resume_text)
            st.markdown(f"""
                <div class="category-badge">
                    <h3 style='margin: 0;'>{category}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Display extracted text with better formatting
            if st.checkbox("Show extracted text", False):
                st.markdown("### 📝 Extracted Resume Text")
                st.text_area("", resume_text, height=300)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Reset progress bar
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            progress_bar.empty()
            status_text.empty()


if __name__ == "__main__":
    main()
