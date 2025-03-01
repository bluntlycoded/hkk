from flask import Flask, request, jsonify
import os
import fitz  # PyMuPDF for PDF text extraction
import re
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Store uploaded filename and analysis globally
uploaded_filename = None
uploaded_analysis = None

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    return text

def extract_references(text):
    references_section = text.split("References")[-1] if "References" in text else ""
    citations = re.findall(r'\[(\d+)\]', references_section)
    return {i: ref for i, ref in enumerate(references_section.split('\n')) if ref.strip()}

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

def together_ai_call(question, context):
    url = "https://api.together.xyz/v1/chat/completions"
    api_key = "7f42f88175b3d9b30542ecf9230ab6c6a970ae03a6c0b2abca2125ad4dc17b15"  # Replace with your actual API key
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are an AI that answers questions based strictly on the given research paper. Even if user asks something outside the paper do not entetain it and simply say that you don't know about it. Also Do not quote anything outside the paper is the most important rule and after every question before answering analyze the paper and answer"},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: No content in response.")
    else:
        return "Error: Failed to get response from Together AI."

def analyze_paper(text):
    summary = "\n".join(text.split("\n")[:10])
    word_count = len(text.split())
    return {"summary": summary, "word_count": word_count}

@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_filename, uploaded_analysis
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    
    text = extract_text_from_pdf(file_path)
    uploaded_filename = file.filename
    uploaded_analysis = analyze_paper(text)
    return jsonify({"message": "File uploaded successfully", "filename": uploaded_filename, "analysis": uploaded_analysis}), 200

@app.route("/check_file", methods=["GET"])
def check_file():
    global uploaded_filename, uploaded_analysis
    if uploaded_filename:
        return jsonify({"filename": uploaded_filename, "analysis": uploaded_analysis}), 200
    else:
        return jsonify({"error": "No file uploaded"}), 404

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    filename = data.get("filename")
    
    if not question or not filename:
        return jsonify({"error": "Missing question or filename"}), 400
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    text = extract_text_from_pdf(file_path)
    references = extract_references(text)
    vector_store = create_vector_store(text)
    
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    answer = together_ai_call(question, context)
    
    citation_match = re.findall(r'\[(\d+)\]', answer)
    citation = references.get(int(citation_match[0]), "No citation found") if citation_match else "No citation found"
    
    return jsonify({"answer": answer, "citation": citation})

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)