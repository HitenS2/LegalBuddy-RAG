from random import randint
from flask import Flask, request, jsonify, render_template, make_response, session
from flask_cors import CORS
import json
import bcrypt
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import jwt

import os
import time
import traceback
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from docx import Document as DocxDocument
import re
from pypdf import PdfReader
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

app = Flask(__name__)
CORS(app, supports_credentials=True)

load_dotenv()

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///legalai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_secret_key')
app.config['JWT_EXPIRATION_DELTA'] = timedelta(days=7)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = db.relationship('Document', backref='owner', lazy=True)
    chat_sessions = db.relationship('ChatSession', backref='user', lazy=True)
    extractions = db.relationship('Extraction', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Document {self.filename}>'

class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationship
    messages = db.relationship('ChatMessage', backref='session', lazy=True)
    
    def __repr__(self):
        return f'<ChatSession {self.id}>'

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_user = db.Column(db.Boolean, default=True)  # True if message from user, False if from AI
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    
    def __repr__(self):
        return f'<ChatMessage {"User" if self.is_user else "AI"}: {self.content[:20]}...>'

class Extraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    content = db.Column(db.JSON, nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<Extraction {self.id}>'

# Create all database tables
with app.app_context():
    db.create_all()

# JWT functions
def generate_token(user_id):
    payload = {
        'exp': datetime.utcnow() + app.config['JWT_EXPIRATION_DELTA'],
        'iat': datetime.utcnow(),
        'sub': user_id
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def decode_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

# Authentication decorator
def token_required(f):
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from Authorization header or cookies
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Authentication token is missing'}), 401
        
        user_id = decode_token(token)
        if not user_id:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user to request context
        current_user = User.query.get(user_id)
        if not current_user:
            return jsonify({'error': 'User not found'}), 404
        
        kwargs['current_user'] = current_user
        return f(*args, **kwargs)
    
    # Preserve the name of the original function
    decorated.__name__ = f.__name__
    return decorated

import subprocess
import os


groq_api_key = os.getenv('GROQ_API_KEY')
groq_api_key_1 = os.getenv('GROQ_API_KEY_1')
groq_api_key_2 = os.getenv('GROQ_API_KEY_2')
groq_api_key_3 = os.getenv('GROQ_API_KEY_3')
groq_api_key_4 = os.getenv('GROQ_API_KEY_4')
groq_api_key_5 = os.getenv('GROQ_API_KEY_5')
groq_api_key_6 = os.getenv('GROQ_API_KEY_6')
groq_api_key_7 = os.getenv('GROQ_API_KEY_7')

google_api_key = os.getenv("GOOGLE_API_KEY")
if not groq_api_key or not google_api_key:
    raise ValueError("Missing required API keys in .env file")

app.config["GOOGLE_API_KEY"] = google_api_key

def convert_docx_to_pdf(docx_path, output_dir):
    """
    Converts a .docx file to .pdf using LibreOffice headless mode
    """
    try:
        subprocess.run([
            'libreoffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', output_dir,
            docx_path
        ], check=True)
        
        base_name = os.path.basename(docx_path).replace('.docx', '.pdf')
        pdf_path = os.path.join(output_dir, base_name)
        
        if os.path.exists(pdf_path):
            return pdf_path
        else:
            raise FileNotFoundError(f"PDF conversion failed: {pdf_path} not found.")
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice conversion failed: {e}")


def extract_dates_from_text(text):
    regex_patterns = [
        # e.g., 01/05/1998 or 01-05-1998
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        # 1st day of May 1998
        r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:day of\s+)?)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
        # May 1st, 1998
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        # May 1998
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}\b"  # Years only
    ]
    matches = []
    for pattern in regex_patterns:
        matches += re.findall(pattern, text, flags=re.IGNORECASE)
    return list(set(matches))


# Initialize LLM with desired parameters
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192",
    temperature=0.7,
    max_tokens=4096
)
llms = [
    ChatGroq(groq_api_key=groq_api_key_1, model_name="Llama3-8b-8192",
             temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_2, model_name="Llama3-8b-8192",
             temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_3, model_name="Llama3-8b-8192",
             temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_4, model_name="Llama3-8b-8192",
                temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_5, model_name="Llama3-8b-8192",
                temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_6, model_name="Llama3-8b-8192",
                temperature=0.7, max_tokens=4096),
    ChatGroq(groq_api_key=groq_api_key_7, model_name="Llama3-8b-8192",
                temperature=0.7, max_tokens=4096)
    
]
groq_api_keys = [groq_api_key_1, groq_api_key_2, groq_api_key_3
                 , groq_api_key_4, groq_api_key_5, groq_api_key_6, groq_api_key_7]


def get_llm(index):
    return llms[index % len(llms)], groq_api_keys[index % len(groq_api_keys)]


# Global storage for processed files and vectors
processed_files = {}
vectors = None
executor = ThreadPoolExecutor(max_workers=4)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv", "xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

# Prompts
LEGAL_EXTRACTION_PROMPT = """You are a precise legal document analyzer. Follow these steps to extract ALL information, including every date, timeline, and related context:

1. First thoroughly analyze the entire document context
2. Extract information in these detailed categories:

ALL DATES AND TIME-RELATED INFORMATION
- List every single date mentioned in the document
- For each date, provide:
  * The exact date
  * Its context (what the date refers to)
  * Related conditions or requirements
  * Any associated deadlines or milestones
- Look for date-related terms like:
  * "as of", "effective", "commencing", "starting"
  * "until", "through", "ending", "terminating"
  * "within", "by", "no later than"
  * "renewal", "extension", "expiration"
  * Any specific day, month, or year mentioned
  Also check system-detected dates provided before the document content and cross-reference them with in-text mentions.


PARTIES AND RELATIONSHIPS
- List every entity or person mentioned
- For each party, include:
  * Their full name/designation
  * Their role in the document
  * Any responsibilities or obligations
  * Relationships with other parties
  * Associated terms or conditions

FINANCIAL DETAILS
- Extract all monetary amounts
- For each financial term:
  * The amount and currency
  * Purpose of payment
  * Payment schedule or timeline
  * Related conditions
  * Associated dates or deadlines

OBLIGATIONS AND REQUIREMENTS
- List all requirements found
- For each obligation:
  * Who is responsible
  * What needs to be done
  * When it needs to be done
  * Related conditions
  * Consequences of non-compliance

LEGAL AND COMPLIANCE
- All legal terms mentioned
- Each compliance requirement
- Regulatory references
- Governing laws
- Jurisdictional details

SPECIAL TERMS
- Identify any unique or special provisions
- Note unusual requirements
- Flag critical conditions
- Highlight important limitations

Format each section with:
1. Main heading
2. Clear bullet points
3. Context for each point
4. Related cross-references

For each extracted item:
- Include the surrounding context
- Note any dependencies or relationships
- Highlight critical implications
- Flag any ambiguities or unclear terms

If any information appears in multiple contexts, list each occurrence with its specific context.
If information is unclear or not specified, state "Not explicitly specified, but related context suggests [explanation]"

Context for analysis:
{context}

Please provide your detailed extraction:"""

SUMMARY_PROMPT = """You are an expert legal document summarizer. Create a concise and clear summary of the provided document following these guidelines:

DOCUMENT OVERVIEW
- Document type and purpose
- Key parties involved
- Overall scope

CORE AGREEMENT TERMS
- Main rights and obligations
- Critical deadlines and dates
- Financial arrangements

KEY LEGAL POINTS
- Governing law and major compliance requirements

NOTABLE PROVISIONS
- Special clauses and limitations

PRACTICAL IMPLICATIONS
- Main takeaways and action items

Ensure your summary is:
- Brief and to the point (around 100–150 words)
- Well-structured with clear headings
- Focused on essential details

Context to summarize:
{context}

Please provide your concise summary:"""

QA_PROMPT = """
You are a legal contract analyst AI. Read the provided contract context and answer the user's question using only specific legal information directly from the document.

🔍 Instructions:
1. Quote all directly relevant clauses using exact wording.
2. Use bullets for multiple related clauses.
3. Keep your **interpretation concise**, avoid repeating the clause itself.
4. Flag any **missing info**, **blank fields**, or **placeholders** (e.g., [____], [XX]).
5. If no answer is available, respond: ❌ Not explicitly stated in the document.
6. Focus on **factual, legal answers only**. Do not generalize or infer beyond the contract text.

📄 Format your response like this:

**Clause(s) Found:**
- "...quoted legal clause here..."
- "...second clause here if applicable..."

**Interpretation:**
- A short explanation (1–2 lines) of what each clause implies.
- Use bullets if multiple clauses apply.
- If fields are blank or placeholders exist, explicitly mention them.

---

User Question:
{input}

Contract Context:
{context}
"""


extraction_prompt = ChatPromptTemplate.from_template(
    f"""{LEGAL_EXTRACTION_PROMPT}

System-detected date hints (pre-extracted by AI engine):
------------------------------
{{date_hints}}

Now thoroughly analyze the document and consider all of the above system-detected dates. Mention if any of these are found in the document context, their purpose, and whether the document context agrees with or adds more detail to them.

Document Content:
------------------------------
{{context}}

Extracted Information:"""
)


def chunk_documents_for_llm(documents, chunk_token_limit=3000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len  # Approx token approximation
    )

    split_docs = text_splitter.split_documents(documents)

    # Combine chunks into safe batches
    current_batch = []
    current_tokens = 0
    batched_docs = []

    for doc in split_docs:
        doc_len = len(doc.page_content)
        if current_tokens + doc_len > chunk_token_limit:
            if current_batch:
                batched_docs.append(current_batch)
            current_batch = [doc]
            current_tokens = doc_len
        else:
            current_batch.append(doc)
            current_tokens += doc_len

    if current_batch:
        batched_docs.append(current_batch)

    return batched_docs


qa_prompt = ChatPromptTemplate.from_template(QA_PROMPT)

# File handling functions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_pdf_text_extractable(file_path):
    try:
        reader = PdfReader(file_path)
        text_content = ""
        for page in reader.pages[:2]:
            text_content += page.extract_text()
        return len(text_content.strip()) > 100
    except Exception as e:
        print(f"PDF verification error: {e}")
        return False


def process_pdf(file_path):
    if is_pdf_text_extractable(file_path):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not any(len(doc.page_content.strip()) > 100 for doc in documents):
                raise ValueError(
                    "Extracted content appears to be insufficient")
            return documents
        except Exception as e:
            print(f"PDF processing error: {e}")
            raise
    else:
        raise ValueError(
            "PDF appears to be image-based or corrupted. Text extraction not supported.")


def process_uploaded_file(file):
    try:
        file_name = file.filename
        file_extension = file_name.rsplit('.', 1)[1].lower()

        if not allowed_file(file_name):
            raise ValueError(f"File type {file_extension} not supported")

        tmp_path = os.path.join(UPLOAD_FOLDER, file_name)
        file.save(tmp_path)

        with open(tmp_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in processed_files:
            print(f"File {file_name} already processed, skipping...")
            return []

        if file_extension == 'pdf':
            documents = process_pdf(tmp_path)
        elif file_extension == 'docx':
    # Convert DOCX to PDF
            pdf_path = convert_docx_to_pdf(tmp_path, output_dir=UPLOAD_FOLDER)
            print(f"Converted DOCX to PDF: {pdf_path}")
            documents = process_pdf(pdf_path)  # Use your existing PDF loader


        elif file_extension == 'txt':
            documents = TextLoader(tmp_path).load()
        elif file_extension == 'csv':
            documents = CSVLoader(tmp_path).load()
        elif file_extension == 'xlsx':
            documents = UnstructuredExcelLoader(tmp_path).load()
        else:
            return []

        if not documents or not any(len(doc.page_content.strip()) > 50 for doc in documents):
            raise ValueError(f"No valid content extracted from {file_name}")

        processed_files[file_hash] = documents
        return documents

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}\n{traceback.format_exc()}")
        raise

# Vectorization with chunking


def vector_embedding(documents):
    global vectors
    print("⚙️ Vector embedding started...")

    try:
        # Load embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Split into token-safe chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False
        )

        final_documents = text_splitter.split_documents(documents)

        if not final_documents:
            raise ValueError(
                "❌ No valid document chunks created after splitting. Aborting embedding.")

        # DEBUG: Log chunk preview
        print(
            f"✅ Ready for embedding. {len(final_documents)} chunks generated.")
        for i, doc in enumerate(final_documents[:5]):
            print(f"[Chunk {i+1} Preview] {doc.page_content[:120]}...")

        # Perform vector embedding
        if vectors is None:
            vectors = FAISS.from_documents(final_documents, embeddings)
        else:
            vectors.add_documents(final_documents)

        print("✅ Vector embedding completed successfully.")
        print(
            f"✅ Uploaded file(s) successfully vectorized into {len(final_documents)} chunks.")

    except Exception as e:
        print(f"❌ Vector embedding error: {str(e)}\n{traceback.format_exc()}")
        raise


@app.route('/')
def index():
    return render_template('index.html')

# Upload endpoint: processes files, creates document chunks, and submits for vectorization


@app.route('/upload', methods=['POST'])
@token_required
def upload_files(current_user):
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400

    all_documents = []
    processed_file_names = []
    uploaded_documents = []
    errors = []
    extracted_text = ""

    for file in files:
        try:
            if file and allowed_file(file.filename):
                file.seek(0)
                file_content = file.read()
                file_hash = hashlib.md5(file_content).hexdigest()
                file.seek(0)

                # Save document record in database
                file_extension = file.filename.rsplit('.', 1)[1].lower()
                
                # Check if the same file has been uploaded before by this user
                existing_doc = Document.query.filter_by(file_hash=file_hash, user_id=current_user.id).first()
                if not existing_doc:
                    # Create new document record
                    new_document = Document(
                        filename=file.filename,
                        file_hash=file_hash,
                        file_type=file_extension,
                        user_id=current_user.id
                    )
                    db.session.add(new_document)
                    db.session.commit()
                    uploaded_documents.append(new_document)
                else:
                    uploaded_documents.append(existing_doc)
                    
                if file_hash not in processed_files:
                    docs = process_uploaded_file(file)
                    all_documents.extend(docs)
                    processed_files[file_hash] = docs
                    processed_file_names.append(file.filename)
                else:
                    print(f"File {file.filename} already processed, skipping...")
        except Exception as e:
            error_msg = f"Error processing {file.filename}: {str(e)}"
            print(f"❌ {error_msg}\n{traceback.format_exc()}")
            errors.append(error_msg)

    if all_documents:
        try:
            global vectors
            vectors = None
            vector_embedding(all_documents)
            extracted_text = "\n".join([
                f"Preview from document chunk {i+1}:\n{doc.page_content[:500]}..."
                for i, doc in enumerate(all_documents[:3])
            ])

        except Exception as e:
            errors.append(f"Error in vector embedding: {str(e)}")

    response = {
        "message": "Files processed successfully ✅" if not errors else "Partial success with errors",
        "processed_files": processed_file_names,
        "document_ids": [doc.id for doc in uploaded_documents],
        "extracted_text": extracted_text,
        "status": "success" if not errors else "partial_success",
        "processed_count": len(processed_file_names)
    }
    if errors:
        response["errors"] = errors

    return jsonify(response)

# Extraction endpoint: returns detailed extraction using the full extraction prompt


@app.route('/extract', methods=['GET'])
@token_required
def extract_details(current_user):
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400
    
    # Get document ID from query parameters (optional)
    document_id = request.args.get('document_id')
    
    # If document ID is provided, verify it belongs to the current user
    if document_id:
        document = Document.query.filter_by(id=document_id, user_id=current_user.id).first()
        if not document:
            return jsonify({"error": "Document not found or access denied"}), 404
    
    try:
        # Maintain sequence as per the image
        section_prompts = {
            "entities": "Extract all party names and their full addresses mentioned in the contract. For each party, include their designation, role in the document, responsibilities, obligations, relationships with other parties, and associated terms/conditions.",
            "dates": "Extract all start and end dates of the contract. Include the context of these dates (what they represent), timeline, milestones, and any associated deadlines. Also cover renewal or termination dates if applicable.",
            "scope": "Extract the scope of work or services mentioned in the contract. Describe what the agreement entails, roles, deliverables, or services.",
            "sla": "Extract SLA (Service Level Agreement) clauses. Include details like expected performance levels, response time, uptime guarantee, metrics, benchmarks, and any service scope limitations. If not present, return '-' symbol.",
            "penalty": "Extract all penalty-related clauses. Mention specific conditions or events that trigger penalties, monetary value of penalties, and enforcement processes.",
            "confidentiality": "Extract confidentiality clauses in the contract. Include information type covered, parties involved, protection duration, and consequence of breach.",
            "termination": "Extract termination and renewal clauses. Specify auto-renewal conditions, exit clause, minimum notice period, early exit provisions, and other termination rules.",
            "commercials": "Extract financial/commercial information from the contract. Include all payment terms, payment amount, frequency, due dates, currency used, billing cycle, and any related conditions.",
            "risks": "Extract all risks, assumptions, special conditions, limitations or disclaimers mentioned in the contract. Highlight uncertainties or dependencies."
        }

        final_output = {}
        total_time = 0.0
        # Track API usage
        api_call_count = 0
        api_key_usage = {}

        retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}
        )

        print("\n📄 Structured Extraction Started...\n")

        def process_section(idx, section, query):
            nonlocal api_call_count, api_key_usage

            max_retries = len(llms)
            for attempt in range(max_retries):
                try:
                    llm_instance, selected_api_key = get_llm((idx + attempt) % len(llms))

                    print(f"🧠 Attempt {attempt + 1}: Groq Key {selected_api_key} for [{section}]")
                    docs = retriever.invoke(query)
                    context = "\n".join(doc.page_content for doc in docs)

                    prompt = f"""You are a legal expert analyzing a contract. Extract the section **{section.upper()}** as per the following instruction:

                    Instruction:
                    {query}

                    Context:
                    {context}

                    ⚠️ Output Formatting Rules:
                    - Use **bold markdown** (`**value**`) for all:
                    - Dates
                    - Party names
                    - Section headings
                    - Monetary amounts
                    - Keep a clean, readable layout with consistent indentation

                    Answer:"""

                    api_call_count += 1
                    api_key_usage[selected_api_key] = api_key_usage.get(selected_api_key, 0) + 1

                    start_time = time.process_time()
                    response = llm_instance.invoke(prompt)
                    elapsed = time.process_time() - start_time

                    print(f"✅ [{section}] Done in {elapsed:.2f}s")
                    return section, str(response.content) if hasattr(response, "content") else str(response), elapsed

                except Exception as e:
                    err_msg = str(e)
                    if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                        print(f"⏳ [{section}] Rate limit hit for {selected_api_key}. Trying next key...")
                        continue  # Try next API key
                    print(f"❌ [{section}] Error: {err_msg}")
                    return section, f"⚠️ Extraction failed: {err_msg}", 0.0

            return section, "❌ All API keys failed or rate-limited", 0.0


        with ThreadPoolExecutor(max_workers=len(section_prompts)) as executor:
            futures = {
                executor.submit(process_section, idx, section, query): section
                for idx, (section, query) in enumerate(section_prompts.items())
            }

            for future in as_completed(futures):
                section, result, elapsed = future.result()
                final_output[section] = result
                total_time += elapsed

        print("✅ All extractions completed successfully.")
        
        # Save extraction to database if document_id is provided
        if document_id:
            extraction = Extraction(
                content=final_output,
                document_id=document_id,
                user_id=current_user.id
            )
            db.session.add(extraction)
            db.session.commit()
            
            extraction_id = extraction.id
        else:
            extraction_id = None
        
        return jsonify({
            "answer": final_output,
            "time_taken": f"{total_time:.2f}s",
            "status": "success",
            "extraction_id": extraction_id,
            "sections_extracted": list(section_prompts.keys()),
            "api_call_summary": {
                "total_calls": api_call_count,
                "key_usage": api_key_usage
            }
        })

    except Exception as e:
        print(f"Batch Extraction Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Batch extraction failed: {str(e)}",
            "status": "error"
        }), 500


@app.route('/dates', methods=['GET'])
def extract_all_dates():
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400

    try:
        retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 15,
                'fetch_k': 30,
                'lambda_mult': 0.7
            }
        )

        raw_docs = retriever.get_relevant_documents("extract dates")
        joined_text = "\n".join(doc.page_content for doc in raw_docs)
        extracted_dates = extract_dates_from_text(joined_text)

        return jsonify({
            "extracted_dates": extracted_dates,
            "date_count": len(extracted_dates),
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": f"Date extraction failed: {str(e)}",
            "status": "error"
        }), 500


# Ask endpoint: processes a legal question based on the uploaded documents
@app.route('/ask', methods=['POST'])
@token_required
def ask_question(current_user):
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400

    data = request.json
    user_question = data.get("query")
    session_id = data.get("session_id")  # Optional, if continuing an existing session

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Get or create chat session
        chat_session = None
        if session_id:
            chat_session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
            if not chat_session:
                return jsonify({"error": "Chat session not found"}), 404
        
        if not chat_session:
            chat_session = ChatSession(user_id=current_user.id)
            db.session.add(chat_session)
            db.session.commit()
        
        # Save user message to database
        user_message = ChatMessage(
            content=user_question,
            is_user=True,
            session_id=chat_session.id
        )
        db.session.add(user_message)
        db.session.commit()

        # Randomly select a Groq LLM
        idx = randint(0, len(llms) - 1)
        llm_instance, selected_api_key = get_llm(idx)
        print(f"🧠 Chatbot using Groq API Key: {selected_api_key}")

        # Use enhanced QA prompt
        qa_prompt_template = ChatPromptTemplate.from_template(QA_PROMPT)
        document_chain = create_stuff_documents_chain(
            llm=llm_instance, prompt=qa_prompt_template)

        # More context for better grounding
        retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 12,
                'fetch_k': 25,
                'lambda_mult': 0.85
            }
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_question})
        elapsed = time.process_time() - start
        
        # Save AI response to database
        ai_message = ChatMessage(
            content=response['answer'],
            is_user=False,
            session_id=chat_session.id
        )
        db.session.add(ai_message)
        db.session.commit()

        return jsonify({
            "answer": response['answer'],
            "time_taken": f"{elapsed:.2f}s",
            "session_id": chat_session.id,
            "message_id": ai_message.id,
            "context_chunks": len(response.get('context', [])),
            "status": "success",
            "api_key_used": selected_api_key[-8:],  # for partial logging
        })

    except Exception as e:
        print(f"❌ Chatbot error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Chatbot failed to process query: {str(e)}",
            "status": "error"
        }), 500

# Get chat history for a user
@app.route('/chat/sessions', methods=['GET'])
@token_required
def get_chat_sessions(current_user):
    sessions = []
    for session in current_user.chat_sessions:
        messages = ChatMessage.query.filter_by(session_id=session.id).order_by(ChatMessage.timestamp).all()
        
        # Get the first user message as a preview
        first_user_message = next((msg for msg in messages if msg.is_user), None)
        preview = first_user_message.content[:50] + "..." if first_user_message else "Empty session"
        
        sessions.append({
            'id': session.id,
            'created_at': session.created_at,
            'message_count': len(messages),
            'preview': preview
        })
    
    return jsonify({
        'sessions': sorted(sessions, key=lambda x: x['created_at'], reverse=True)
    })

# Get messages for a specific chat session
@app.route('/chat/sessions/<int:session_id>/messages', methods=['GET'])
@token_required
def get_chat_messages(current_user, session_id):
    # Verify session belongs to user
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return jsonify({"error": "Chat session not found"}), 404
    
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
    message_list = [{
        'id': msg.id,
        'content': msg.content,
        'timestamp': msg.timestamp,
        'is_user': msg.is_user
    } for msg in messages]
    
    return jsonify({
        'session_id': session_id,
        'messages': message_list
    })

@app.route('/summary', methods=['POST'])
def ask_summary():
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400

    try:
        summary_prompt_template = ChatPromptTemplate.from_template(
            SUMMARY_PROMPT)

        # Create document chain with the updated prompt
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=summary_prompt_template,
        )

        # Use reduced retrieval parameters to lower token usage
        retriever = vectors.as_retriever(
            search_type="similarity",
            search_kwargs={
                'k': 5,           # Fewer, more focused chunks
                'fetch_k': 15,    # Limit token consumption
                'lambda_mult': 0.7,
                'filter': None
            }
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({
            'input': 'Provide a concise summary of the document focusing on key points'
        })
        elapsed = time.process_time() - start

        return jsonify({
            "answer": response['answer'],
            "time_taken": f"{elapsed:.2f}s",
            "context_chunks": len(response.get('context', [])),
            "status": "success"
        })

    except Exception as e:
        print(f"Summary error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Summary generation failed: {str(e)}",
            "status": "error"
        }), 500

# Helper function to generate PDF
def generate_pdf(title, content, is_extraction=False):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom styles - using unique names to avoid conflicts
    title_style = ParagraphStyle(
        name='DocTitle', 
        parent=styles['Heading1'], 
        fontSize=16, 
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    section_style = ParagraphStyle(
        name='SectionHeader', 
        parent=styles['Heading2'], 
        fontSize=14, 
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue
    )
    
    content_style = ParagraphStyle(
        name='BodyContent', 
        parent=styles['Normal'], 
        fontSize=11, 
        alignment=TA_LEFT,
        spaceAfter=12,
        leading=14
    )
    
    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))
    
    if is_extraction:
        # For extraction results - structured format with sections
        section_order = [
            ("entities", "1) Entities Name & Address Details"),
            ("dates", "2) Contract Start Date & End Date"),
            ("scope", "3) Scope"),
            ("sla", "4) SLA Clause"),
            ("penalty", "5) Penalty Clause"),
            ("confidentiality", "6) Confidentiality Clause"),
            ("termination", "7) Renewal and Termination Clause"),
            ("commercials", "8) Commercials / Payment Terms"),
            ("risks", "9) Risks / Assumptions")
        ]
        
        for key, section_title in section_order:
            if key in content:
                elements.append(Paragraph(section_title, section_style))
                
                # Process markdown-style bold text for PDF
                text = content[key]
                # Replace markdown bold with HTML tags
                processed_text = text.replace('**', '<b>', 1)
                while '**' in processed_text:
                    processed_text = processed_text.replace('**', '</b>', 1).replace('**', '<b>', 1)
                # Ensure all bold tags are closed
                if processed_text.count('<b>') > processed_text.count('</b>'):
                    processed_text += '</b>'
                
                # Handle bullet points
                lines = processed_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- ') or line.startswith('* '):
                        # Indent bullet points properly
                        line = '• ' + line[2:]
                        elements.append(Paragraph(line, content_style))
                    elif line:
                        elements.append(Paragraph(line, content_style))
                
                elements.append(Spacer(1, 6))
    else:
        # For chatbot/QA responses - simple format
        elements.append(Paragraph("<b>Question:</b>", section_style))
        elements.append(Paragraph(content.split("\n\nAnswer:")[0].replace("Question:\n", ""), content_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("<b>Answer:</b>", section_style))
        answer_text = content.split("\n\nAnswer:\n")[-1]
        
        # Process markdown-style bold text
        processed_text = answer_text.replace('**', '<b>', 1)
        while '**' in processed_text:
            processed_text = processed_text.replace('**', '</b>', 1).replace('**', '<b>', 1)
        if processed_text.count('<b>') > processed_text.count('</b>'):
            processed_text += '</b>'
        
        # Handle bullet points and line breaks properly
        lines = processed_text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Add space between paragraphs
                if i > 0 and i < len(lines) - 1:
                    elements.append(Spacer(1, 6))
                continue
            
            # Handle Clause(s) Found and Interpretation sections
            if line.startswith('**Clause') or line.startswith('**Interpretation'):
                elements.append(Paragraph(line.replace('**', '<b>').replace('**', '</b>'), section_style))
                continue
                
            # Handle bullet points
            if line.startswith('- '):
                line = '• ' + line[2:]
                # Create indented style for bullets
                bullet_style = ParagraphStyle(
                    name='BulletStyle',
                    parent=content_style,
                    leftIndent=20
                )
                elements.append(Paragraph(line, bullet_style))
            else:
                elements.append(Paragraph(line, content_style))
            
            # Add small space between lines if not a bullet point
            if i < len(lines) - 1 and not lines[i+1].startswith('- '):
                elements.append(Spacer(1, 2))
    
    # Add a footer with date
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        name='Footer', 
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )
    elements.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# Add PDF download endpoints
@app.route('/download/extraction', methods=['POST'])
def download_extraction():
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400
    
    data = request.json
    extraction_data = data.get("extraction")
    
    if not extraction_data:
        return jsonify({"error": "No extraction data provided"}), 400
    
    try:
        pdf = generate_pdf("Contract Extraction Report", extraction_data, is_extraction=True)
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=contract_extraction.pdf'
        return response
        
    except Exception as e:
        print(f"PDF Generation Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"PDF generation failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/download/chat', methods=['POST'])
def download_chat():
    if not vectors:
        return jsonify({"error": "No documents uploaded"}), 400
    
    data = request.json
    question = data.get("question", "")
    answer = data.get("answer", "")
    
    if not answer:
        return jsonify({"error": "No answer provided"}), 400
    
    try:
        content = f"Question:\n{question}\n\nAnswer:\n{answer}"
        pdf = generate_pdf("Contract Q&A", content)
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=contract_qa.pdf'
        return response
        
    except Exception as e:
        print(f"PDF Generation Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"PDF generation failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/download/chat-history', methods=['POST'])
@token_required
def download_chat_history(current_user):
    data = request.json
    session_id = data.get("session_id")
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    # Verify session belongs to user
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return jsonify({"error": "Chat session not found"}), 404
    
    # Get all messages in the session
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
    if not messages:
        return jsonify({"error": "No messages found in this session"}), 404
    
    try:
        # Format the chat history content
        chat_content = ""
        for i, msg in enumerate(messages):
            if msg.is_user:
                chat_content += f"Question {i//2 + 1}:\n{msg.content}\n\n"
            else:
                chat_content += f"Answer {i//2 + 1}:\n{msg.content}\n\n"
                if i < len(messages) - 1:  # Add separator between Q&A pairs
                    chat_content += "---\n\n"
        
        # Find document name if available
        document_name = "Unknown"
        if session.id:
            # Try to find a document linked to any extraction made by this user
            extraction = Extraction.query.filter_by(user_id=current_user.id).first()
            if extraction:
                document = Document.query.get(extraction.document_id)
                if document:
                    document_name = document.filename
        
        # Generate PDF with the chat history
        pdf = generate_pdf_chat_history(f"Chat History: {document_name}", chat_content)
        
        # Create response with the PDF
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=chat_history_{session_id}.pdf'
        return response
        
    except Exception as e:
        print(f"PDF Generation Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": f"PDF generation failed: {str(e)}",
            "status": "error"
        }), 500

# Helper function to generate PDF for chat history
def generate_pdf_chat_history(title, content):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='DocTitle', 
        parent=styles['Heading1'], 
        fontSize=16, 
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    question_style = ParagraphStyle(
        name='Question', 
        parent=styles['Heading2'], 
        fontSize=14, 
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue
    )
    
    answer_style = ParagraphStyle(
        name='Answer', 
        parent=styles['Heading2'], 
        fontSize=14, 
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkgreen
    )
    
    content_style = ParagraphStyle(
        name='Content', 
        parent=styles['Normal'], 
        fontSize=11, 
        alignment=TA_LEFT,
        spaceAfter=12,
        leading=14
    )
    
    separator_style = ParagraphStyle(
        name='Separator',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=TA_CENTER
    )
    
    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 12))
    
    # Chat history parsing
    sections = content.split("---\n\n")
    
    for i, section in enumerate(sections):
        qa_parts = section.split("\n\n", 1)
        if len(qa_parts) < 2:
            continue
            
        question_part = qa_parts[0]
        answer_part = qa_parts[1]
        
        # Process question
        if question_part.startswith("Question "):
            question_header = question_part.split(":\n", 1)[0] + ":"
            question_content = question_part.split(":\n", 1)[1]
            
            elements.append(Paragraph(question_header, question_style))
            
            # Process question content
            lines = question_content.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    elements.append(Paragraph(line, content_style))
        
        # Process answer
        if answer_part.startswith("Answer "):
            answer_header = answer_part.split(":\n", 1)[0] + ":"
            answer_content = answer_part.split(":\n", 1)[1]
            
            elements.append(Paragraph(answer_header, answer_style))
            
            # Process markdown-style formatting in answer
            processed_text = answer_content.replace('**', '<b>', 1)
            while '**' in processed_text:
                processed_text = processed_text.replace('**', '</b>', 1).replace('**', '<b>', 1)
            if processed_text.count('<b>') > processed_text.count('</b>'):
                processed_text += '</b>'
            
            # Process answer content with proper line breaks and bullets
            lines = processed_text.split('\n')
            for j, line in enumerate(lines):
                line = line.strip()
                if not line:
                    if j > 0 and j < len(lines) - 1:
                        elements.append(Spacer(1, 6))
                    continue
                
                # Handle Clause(s) Found and Interpretation sections
                if line.startswith('**Clause') or line.startswith('**Interpretation'):
                    elements.append(Paragraph(line.replace('**', '<b>').replace('**', '</b>'), question_style))
                    continue
                    
                # Handle bullet points
                if line.startswith('- '):
                    line = '• ' + line[2:]
                    # Create indented style for bullets
                    bullet_style = ParagraphStyle(
                        name='BulletStyle',
                        parent=content_style,
                        leftIndent=20
                    )
                    elements.append(Paragraph(line, bullet_style))
                else:
                    elements.append(Paragraph(line, content_style))
        
        # Add separator between Q&A pairs except for the last one
        if i < len(sections) - 1:
            elements.append(Paragraph("* * *", separator_style))
            elements.append(Spacer(1, 12))
    
    # Add a footer with date
    elements.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        name='Footer', 
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )
    elements.append(Paragraph(f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# User registration
@app.route('/auth/register', methods=['POST'])
def register():
    data = request.json
    
    # Validate input data
    if not data or not all(k in data for k in ('email', 'password', 'name')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if user already exists
    existing_user = User.query.filter_by(email=data['email']).first()
    if existing_user:
        return jsonify({'error': 'Email already registered'}), 409
    
    # Hash password
    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create new user
    new_user = User(
        email=data['email'],
        password=hashed_password,
        name=data['name']
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    # Generate token
    token = generate_token(new_user.id)
    
    return jsonify({
        'token': token,
        'user': {
            'id': new_user.id,
            'email': new_user.email,
            'name': new_user.name
        }
    }), 201

# User login
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    
    # Validate input data
    if not data or not all(k in data for k in ('email', 'password')):
        return jsonify({'error': 'Missing email or password'}), 400
    
    # Find user
    user = User.query.filter_by(email=data['email']).first()
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    # Generate token
    token = generate_token(user.id)
    
    return jsonify({
        'token': token,
        'user': {
            'id': user.id,
            'email': user.email,
            'name': user.name
        }
    })

# Get current user profile
@app.route('/auth/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    return jsonify({
        'user': {
            'id': current_user.id,
            'email': current_user.email,
            'name': current_user.name,
            'created_at': current_user.created_at
        }
    })

# Get user's documents and extractions
@app.route('/user/documents', methods=['GET'])
@token_required
def get_user_documents(current_user):
    documents = []
    for doc in current_user.documents:
        extractions = Extraction.query.filter_by(document_id=doc.id).all()
        has_extraction = len(extractions) > 0
        
        documents.append({
            'id': doc.id,
            'filename': doc.filename,
            'uploaded_at': doc.uploaded_at,
            'file_type': doc.file_type,
            'has_extraction': has_extraction
        })
    
    return jsonify({
        'documents': documents
    })

# Get all extractions for a user
@app.route('/extractions', methods=['GET'])
@token_required
def get_user_extractions(current_user):
    extractions = []
    extraction_records = Extraction.query.filter_by(user_id=current_user.id).order_by(Extraction.created_at.desc()).all()
    
    for extraction in extraction_records:
        document = Document.query.get(extraction.document_id)
        if document:
            extractions.append({
                'id': extraction.id,
                'created_at': extraction.created_at,
                'document_id': extraction.document_id,
                'document_name': document.filename
            })
    
    return jsonify({
        'extractions': extractions
    })

# Get a specific extraction
@app.route('/extractions/<int:extraction_id>', methods=['GET'])
@token_required
def get_extraction(current_user, extraction_id):
    extraction = Extraction.query.filter_by(id=extraction_id, user_id=current_user.id).first()
    if not extraction:
        return jsonify({"error": "Extraction not found"}), 404
    
    document = Document.query.get(extraction.document_id)
    
    return jsonify({
        'id': extraction.id,
        'created_at': extraction.created_at,
        'document_id': extraction.document_id,
        'document_name': document.filename if document else "Unknown",
        'content': extraction.content
    })

# Delete a specific extraction
@app.route('/extractions/<int:extraction_id>', methods=['DELETE'])
@token_required
def delete_extraction(current_user, extraction_id):
    extraction = Extraction.query.filter_by(id=extraction_id, user_id=current_user.id).first()
    if not extraction:
        return jsonify({"error": "Extraction not found"}), 404
    
    db.session.delete(extraction)
    db.session.commit()
    
    return jsonify({
        'message': 'Extraction deleted successfully',
        'id': extraction_id
    })

# Delete a chat session and all its messages
@app.route('/chat/sessions/<int:session_id>', methods=['DELETE'])
@token_required
def delete_chat_session(current_user, session_id):
    session = ChatSession.query.filter_by(id=session_id, user_id=current_user.id).first()
    if not session:
        return jsonify({"error": "Chat session not found"}), 404
    
    # Delete all messages in the session
    ChatMessage.query.filter_by(session_id=session_id).delete()
    
    # Delete the session
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({
        'message': 'Chat session deleted successfully',
        'id': session_id
    })

if __name__ == '__main__':
    app.run(debug=False)
