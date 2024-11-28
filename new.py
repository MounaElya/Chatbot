import PyPDF2  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

pytesseract.pytesseract.tesseract_cmd=r"C:\\Users\\asus\\Downloads\\tesseract-ocr-w64-setup-5.4.0.20240606.exe"
# Extract text from pdf
def extract_text_from_pdf_with_ocr(pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
            #Extract text directly if possible
                page_text= page.extract_text()
                if page_text:
                    text+=page_text+"\n"
                else:
                    #If not, use OCR to extract text
                    image=page.to_image().original
                    ocr_text=pytesseract.image_to_string(Image.fromarray(image))
                    text+=ocr_text+"\n"
        return text

# Download necessary resources for nltk
nltk.download('punkt')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip().lower()
    return text

def extract_qa_pairs(text):
    qa_pairs = re.split(r'\d', text)
    qa_pairs = [qa.strip() for qa in qa_pairs if qa.strip()]
    return qa_pairs

def split_questions_answers(qa_pairs):
    qa_dict = {}
    for pair in qa_pairs:
        if 'answer' in pair:
            question, answer = pair.split('answer', 1)
            qa_dict[question.strip()] = answer.strip()
    return qa_dict

def vectorize_question(questions):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)
    return vectorizer, vectors


def find_best_match(query, vectorizer, question_vectors, questions):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, question_vectors)
    best_match_idx = similarities.argmax()
    best_question = questions[best_match_idx]
    
    # If the similarity score is below a threshold, use fuzzy matching
    if similarities[0][best_match_idx] < 0.5:
        best_question, _ = process.extractOne(query, questions)
    
    return best_question

def get_answer_for_query(query, vectorizer, question_vectors, questions, qa_dict):
    best_question = find_best_match(query, vectorizer, question_vectors, questions)
    return qa_dict[best_question]

def load_faq_data(pdf_path):
    extracted_text = extract_text_from_pdf_with_ocr(pdf_path)
    cleaned_text = clean_text(extracted_text)

    qa_pairs = extract_qa_pairs(cleaned_text)
    qa_dict = split_questions_answers(qa_pairs)

    questions = list(qa_dict.keys())
    vectorizer, question_vectors = vectorize_question(questions)
    
    return vectorizer, question_vectors, questions, qa_dict
