import streamlit as st
import pickle
import string
import nltk
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Setup
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()

# --- Text Preprocessing ---
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load Model & Vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Streamlit Page Config ---
st.set_page_config(page_title="🔥 Spam Detector", page_icon="📨", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #dff0ff, #fdfbfb);
        }
        .title {
            font-size: 3em;
            text-align: center;
            font-weight: bold;
            color: #2b4162;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.1em;
            margin-bottom: 2rem;
        }
        .left-column {
            padding: 2rem;
        }
        .output-box {
            margin-top: 3rem;
            font-size: 1.5rem;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        .spam {
            background-color: #ffe6e6;
            color: #d11a2a;
        }
        .not-spam {
            background-color: #e6ffe6;
            color: #2e8b57;
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 0.9rem;
            margin-top: 3rem;
        }
        .custom-textarea textarea {
            border: 2px solid #4CAF50 !important;
            border-radius: 12px !important;
            box-shadow: 0px 5px 20px rgba(76, 175, 80, 0.1);
            font-size: 16px !important;
            padding: 1rem !important;
            background-color: #ffffff !important;
            transition: all 0.3s ease-in-out;
        }
        div.stButton > button {
        height: 50px;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 18px;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
        border: none;
        box-shadow: 0px 4px 14px rgba(0, 0, 0, 0.1);
        }

        div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.15);
        }
        
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='title'>📨 Check Spam Email</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect if a message is Spam or Not using Machine Learning & NLP magic!</div>", unsafe_allow_html=True)

st.markdown("#### ✍️ Write or paste your message below:")
input_sms = st.text_area("", placeholder="Enter Your Message", height=180, key="message", label_visibility="collapsed")


# --- Prediction Logic ---
predict_container = st.empty()

if st.button("🚀 Submit"):
    if not input_sms.strip():
        st.warning("Please enter a message to check.")
    else:
        # Typing animation
        with predict_container:
            with st.spinner("Analyzing..."):
                time.sleep(1.2)  # Simulate loading

        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Fade-in animation styling
        st.markdown("""
            <style>
            @keyframes fadeIn {
                0% {opacity: 0; transform: translateY(10px);}
                100% {opacity: 1; transform: translateY(0);}
            }
            .fade-result {
                animation: fadeIn 1s ease-out;
                text-align: center;
                font-size: 1.6rem;
                padding: 2rem;
                border-radius: 12px;
                margin-top: 2rem;
            }
            .spam {
                background-color: #ffe6e6;
                color: #d11a2a;
                box-shadow: 0 0 10px #f8d7da;
            }
            .not-spam {
                background-color: #e6ffe6;
                color: #2e8b57;
                box-shadow: 0 0 10px #c1f5c1;
            }
            </style>
        """, unsafe_allow_html=True)

        # Final Result Display
        if result == 1:
            st.markdown(
                '<div class="fade-result spam">🚫 <strong>SPAM DETECTED!</strong> <br> This message looks suspicious! ⚠️</div>',
                unsafe_allow_html=True
            )
        else:
            #st.balloons()
            st.markdown(
                '<div class="fade-result not-spam">✅ <strong>All Clear! Not Spam</strong> <br> This message is safe! 🎉</div>',
                unsafe_allow_html=True
            )


# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & NLTK</div>", unsafe_allow_html=True)
