from http.client import HTTPException
from bson.binary import Binary  # type: ignore # Add this import instead
from google.oauth2.credentials import Credentials # type: ignore
from googleapiclient.discovery import build # type: ignore
from google.auth.transport.requests import Request # type: ignore
from fastapi import FastAPI, Depends, HTTPException, status # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Any, Collection, List, Dict, Union
import httpx
from pydantic import BaseModel # type: ignore
from pymongo import MongoClient  # type: ignore # Import MongoClient for MongoDB connection
from bs4 import BeautifulSoup # type: ignore
from sentence_transformers import SentenceTransformer, util
from typing import Tuple
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError as JWTError
import base64
import re
import os
import email  # Import this to use email.message_from_bytes
from pymongo.server_api import ServerApi # type: ignore
import fitz  # type: ignore # PyMuPDF for PDF handling
from docx import Document # type: ignore
from io import BytesIO
from app.process_cv import extract_text_from_pdf, match_job_roles, job_role_keywords
import json
import re
import uuid
import os
import chardet # type: ignore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dateutil.relativedelta import relativedelta
from email.mime.base import MIMEBase
from email import encoders
from fastapi import File, Form, UploadFile
import shutil
from email.mime.text import MIMEText
import json  # For parsing lists in working_mode and expected_role
from datetime import datetime,timezone
from fastapi.responses import Response

from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()


# MongoDB connection URI from environment variable
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "sltinterndata")

# Create a new client and connect to MongoDB
client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))

# Get CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test the connection to MongoDB
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Define the database and collections
db = client[MONGODB_DATABASE]
collection = db['internsdata']
cv_collection = db['cvs']  # New collection for storing CV pdfs 
cvText_collection = db['cvText']  
users_collection = db["users"]
shortlist_collection = db["shortlist"]
hired_interns_collection = db["hired_interns"]

# Load the pre-trained Sentence Transformer model
# Load the pre-trained Sentence Transformer model
model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
encoder = SentenceTransformer(model_name)

data = []

# First, update the EmailData model to include PDF data
class EmailData(BaseModel):
    name: str
    email: str
    degree: str
    university: str
    current_year: str
    internship_period: str
    working_mode: List[str]
    expected_role: List[str]
    starting_date: str
    contact_no: str
    skills: Dict[str, Union[List[str], str]]
    statement: str
    cv_link: str
    uploaded_time: str = None
    cv_id: str = None
    # Add these fields to store PDF data
    pdf_extracted_text: str = None
    pdf_possible_roles: List[str] = None
    pdf_filename: str = None


# Gmail API credentials from environment variables
# ACCESS_TOKEN = os.getenv("GMAIL_ACCESS_TOKEN")
# REFRESH_TOKEN = os.getenv("GMAIL_REFRESH_TOKEN")
# CLIENT_ID = os.getenv("GMAIL_CLIENT_ID")
# CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET")
# TOKEN_URI = os.getenv("GMAIL_TOKEN_URI", "https://oauth2.googleapis.com/token")

#N8N configurations
N8N_URL = os.getenv("N8N_WORKFLOW_WEBHOOK_URL")

# Email configuration from environment variables
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# File storage configuration from environment variables
ATTACHMENTS_DIR = os.getenv("ATTACHMENTS_DIR", "attachments")
DEFAULT_ATTACHMENT_FILENAME = os.getenv("DEFAULT_ATTACHMENT_FILENAME", "trainee_guidelines.pdf")
DEFAULT_ATTACHMENT_PATH = os.path.join(ATTACHMENTS_DIR, DEFAULT_ATTACHMENT_FILENAME)

# Create attachments directory if it doesn't exist
if not os.path.exists(ATTACHMENTS_DIR):
    os.makedirs(ATTACHMENTS_DIR)
    print(f"Created directory: {ATTACHMENTS_DIR}")

# Copy the default attachment on startup if needed
def initialize_default_attachment():
    """Initialize the default attachment from the provided PDF file."""
    # Define the path to the source PDF (assuming it's in the same directory as main.py)
    source_pdf = DEFAULT_ATTACHMENT_FILENAME  # Update this path to match your PDF location
    
    if os.path.exists(source_pdf) and not os.path.exists(DEFAULT_ATTACHMENT_PATH):
        try:
            shutil.copy(source_pdf, DEFAULT_ATTACHMENT_PATH)
            print(f"Default attachment copied to: {DEFAULT_ATTACHMENT_PATH}")
        except Exception as e:
            print(f"Error copying default attachment: {e}")
    elif not os.path.exists(source_pdf):
        print(f"Warning: Default attachment source file not found at {source_pdf}")

# Call this function during app startup
initialize_default_attachment()

# Function to authenticate Gmail API using the provided tokens
# def authenticate_gmail():
#     creds = Credentials(
#         token=ACCESS_TOKEN,
#         refresh_token=REFRESH_TOKEN,
#         client_id=CLIENT_ID,
#         client_secret=CLIENT_SECRET,
#         token_uri=TOKEN_URI
#     )

#     if not creds.valid:
#         if creds.expired and creds.refresh_token:
#             creds.refresh(Request())

#     return creds

def _pick_cv_attachment(attachments: list) -> dict:
    """Pick a CV attachment from the list of attachments."""
    if not attachments:
        return None
    
    for att in attachments:
        filename = (att.get("filename") or "").lower()
        mime = (att.get("mimeType") or "").lower()
        
        # Check if it's a PDF or common CV format
        if filename.endswith((".pdf", ".doc", ".docx")) or mime in ("application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            return att
    
    # If no PDF found, return the first attachment
    return attachments[0] if attachments else None

def _safe_email_from_from_field(from_field: str) -> str:
    """Extract email address from the From field."""
    if not from_field:
        return ""
    
    # Try to extract email from format: "Name <email@domain.com>"
    match = re.search(r"<(.+?)>", from_field)
    if match:
        return match.group(1).strip()
    
    # If no angle brackets, assume the whole thing is an email
    return from_field.strip()

def _parse_isoish_date(date_str: str) -> datetime:
    """Parse ISO-like date strings."""
    if not date_str:
        return None
    
    try:
        # Try ISO format first
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pass
    
    try:
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%a, %b %d, %Y %H:%M:%S %z"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    except Exception:
        pass
    
    return None

def decode_email_raw(raw):
    """Decode the raw email content."""
    message_bytes = base64.urlsafe_b64decode(raw)
    message = email.message_from_bytes(message_bytes)

    def decode_payload(payload):
        try:
            # Try decoding as UTF-8
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            # Detect encoding and decode accordingly
            detected_encoding = chardet.detect(payload).get("encoding", "latin1")
            return payload.decode(detected_encoding, errors="replace")

    # If the email has multiple parts, extract text content
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":  # Prefer plain text over HTML
                return decode_payload(part.get_payload(decode=True))
            elif content_type == "text/html":  # Use HTML only if plain text is unavailable
                html_body = decode_payload(part.get_payload(decode=True))
                return BeautifulSoup(html_body, "html.parser").get_text()
    else:
        # Single-part email
        return decode_payload(message.get_payload(decode=True))

    return ""  # Return empty if no suitable content is found


def extract_skills(text: str):

    if not text:
        return {}

    # Capture Skills block until the next known section label or end of text
    block_match = re.search(
        r"Skills:\s*(.*?)(?:\n(?:Statement|CV|Contact No|Starting Date|Expected Role|Working Mode|Internship Period|Current Year|Institute|Course|Email|Name):|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return {}

    skills_block = block_match.group(1).strip()
    if not skills_block:
        return {}

    skills_dict = {}

    pair_matches = re.findall(
        r"([A-Za-z][A-Za-z0-9 &/\-]+?)\s*-\s*(\[[^\]]*\])",
        skills_block,
        flags=re.DOTALL,
    )

    for category, skill_list in pair_matches:
        category = category.strip()
        raw = skill_list.strip()

        # Normalize quotes so json.loads can parse
        normalized = raw.replace("'", '"')

        try:
            parsed = json.loads(normalized)
            # Ensure list
            if isinstance(parsed, list):
                skills_dict[category] = parsed
            else:
                skills_dict[category] = [str(parsed)]
        except json.JSONDecodeError:
            # Fallback: try to split manually if it's not valid JSON
            inner = raw.strip()[1:-1].strip()
            if inner:
                skills_dict[category] = [s.strip().strip('"').strip("'") for s in inner.split(",") if s.strip()]
            else:
                skills_dict[category] = []

    # If it didn't match "Category-[...]" format, fallback: treat as comma-separated list
    if not skills_dict:
        items = [s.strip() for s in re.split(r"[,\n•]+", skills_block) if s.strip()]
        if items:
            skills_dict["Skills"] = items

    return skills_dict
  # Convert dict back to JSON string for parsing

CV_STORE = {}  # cv_id -> { "bytes": ..., "mime": ..., "filename": ... }


@app.get("/cv/{cv_id}")
def get_cv(cv_id: str):
    cv = CV_STORE.get(cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")
    return Response(
        content=cv["bytes"],
        media_type=cv["mime"],
        headers={"Content-Disposition": f'inline; filename="{cv["filename"]}"'}
    )  


async def fetch_recent_emails(since_time: Optional[str] = None) -> List[Dict[str, Any]]:
    N8N_URL = os.getenv("N8N_WORKFLOW_WEBHOOK_URL")
    if not N8N_URL:
        raise RuntimeError("Missing env var: N8N_WORKFLOW_WEBHOOK_URL")

    params = {}
    if since_time:
        params["since_time"] = since_time

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.get(N8N_URL, params=params)
        resp.raise_for_status()
        items = resp.json()

    emails = []

    for item in items:
        try:
            body_data = item.get("body_text", "") or ""

            def safe_regex(pattern, text, group=1):
                match = re.search(pattern, text, re.MULTILINE)
                return match.group(group).strip() if match else ""

            def safe_json_parse(text):

                # Already parsed
                if isinstance(text, (list, dict)):
                    return text

                # None or empty
                if not text or not isinstance(text, str):
                    return []

                text = text.strip()
                if not text:
                    return []

                # Try JSON first
                try:
                    return json.loads(text.replace("'", '"'))
                except json.JSONDecodeError:
                     pass

                # Fallback: treat as single value list
                return [text]


            email_data = {
                "name": safe_regex(r"Name:\s*(.*)", body_data),
                "email": safe_regex(r"Email:\s*(.*)", body_data),
                "degree": safe_regex(r"Course:\s*(.*)", body_data),
                "university": safe_regex(r"Institute:\s*(.*)", body_data),
                "current_year": safe_regex(r"Current Year:\s*(.*)", body_data),
                "internship_period": safe_regex(r"Internship Period:\s*(.*)", body_data),
                "working_mode": safe_json_parse(safe_regex(r"Working Mode:\s*(\[.*\])", body_data)),
                "expected_role": safe_json_parse(safe_regex(r"Expected Role:\s*(\[.*\])", body_data)),
                "starting_date": safe_regex(r"Starting Date:\s*(.*)", body_data),
                "contact_no": safe_regex(r"Contact No:\s*(.*)", body_data),
                "skills": extract_skills(body_data),
                "statement": safe_regex(r"Statement:\s*(.*)", body_data),
                "cv_link": "",
            }

            # --- attachment handling from n8n ---
            attachments = item.get("attachments") or []
            # pick first pdf
            pdf_att = None
            for a in attachments:
                fn = (a.get("filename") or "").lower()
                mt = (a.get("mimeType") or "").lower()
                if fn.endswith(".pdf") or mt == "application/pdf":
                    pdf_att = a
                    break

            if pdf_att and pdf_att.get("contentBase64"):
                cv_id = str(uuid.uuid4())
                pdf_bytes = base64.b64decode(pdf_att["contentBase64"])
                CV_STORE[cv_id] = {
                    "bytes": pdf_bytes,
                    "mime": pdf_att.get("mimeType") or "application/pdf",
                    "filename": pdf_att.get("filename") or f"{cv_id}.pdf",
                }
                email_data["cv_id"] = cv_id
                email_data["cv_link"] = f"http://127.0.0.1:5001/cv/{cv_id}"  # change host/port if needed
            else:
                email_data["cv_link"] = ""

            # ---- cleanup (your same logic) ----
            if not isinstance(email_data.get("skills"), (list, dict)):
                email_data["skills"] = []

            email_data["working_mode"] = [str(x) for x in (email_data["working_mode"] or [])]
            email_data["expected_role"] = [str(x) for x in (email_data["expected_role"] or [])]

            email_data = {k: (v.strip() if isinstance(v, str) else v) for k, v in email_data.items()}

            emails.append(email_data)

        except Exception as e:
            print(f"Error processing n8n item: {e}")
            continue

    return emails


@app.get("/emails")
async def get_emails():
    return await fetch_recent_emails()

#--------------------------------------------------------------------------------------------------
    
@app.get("/emails_since_last_upload")
def get_emails_since_last_upload():
    try:
        # Fetch the latest uploaded time
        latest_record = collection.find_one(sort=[("uploaded_time", -1)])
        last_uploaded_time = latest_record.get("uploaded_time") if latest_record else None
        print(f"Last uploaded time: {last_uploaded_time}")

        # Fetch emails received after the last uploaded time
        emails = fetch_recent_emails()
        return {"emails": emails}

    except Exception as e:
        print(f"Error in emails_since_last_upload: {e}")
        return HTTPException(500, f"An error occurred: {str(e)}")


def separate_cv_data(email_data_dict: dict) -> Tuple[dict, dict]:
    """
    Separates CV-related data from email data and returns both separately.
    Keeps cv_id in both collections for relationship mapping.
    """
    cv_fields = {
        'pdf_extracted_text',
        'pdf_possible_roles',
        'pdf_filename'
    }
    
    # Extract CV data
    cv_data = {
        'cv_id': email_data_dict.get('cv_id'),  # This will be in both collections
        'candidate_name': email_data_dict.get('name'),
        'candidate_email': email_data_dict.get('email'),
        'extracted_text': email_data_dict.get('pdf_extracted_text'),
        'possible_roles': email_data_dict.get('pdf_possible_roles') or [],
        'upload_date': datetime.now(),
        'original_filename': email_data_dict.get('pdf_filename')
    }
    
    # Remove CV-related fields from email data, but keep cv_id
    email_data = {k: v for k, v in email_data_dict.items() if k not in cv_fields}
    
    return email_data, cv_data

@app.post("/save_email_data")
async def save_email_data(email_data: EmailData):
    try:
        print(f"\nSaving email data for: {email_data.name}")
        
        # Convert the Pydantic model to a dictionary
        email_data_dict = email_data.dict()
        email_data_dict["uploaded_time"] = datetime.now().isoformat()
        
        # Separate CV data from email data
        clean_email_data, cv_data = separate_cv_data(email_data_dict)
        
        # Save the clean email data to MongoDB
        collection.insert_one(clean_email_data)
        print("Saved email data to main collection")
        
        # If we have CV data, save it to the cvText collection
        if cv_data['cv_id'] and cv_data['extracted_text']:
            cvText_collection.insert_one(cv_data)
            print(f"Successfully saved CV text to cvText collection with ID: {cv_data['cv_id']}")
            
            # Print the extracted text
            print("\nExtracted CV Text Preview:")
            print("=" * 50)
            preview_text = cv_data['extracted_text'][:500] + "..." if cv_data['extracted_text'] else "No text extracted"
            print(preview_text)
            print("=" * 50)
            
        return {
            "success": True,
            "message": "Email data saved successfully",
            "cv_id": cv_data['cv_id'],
            "pdf_processed": bool(cv_data['extracted_text'])
        }
        
    except Exception as e:
        print(f"Error in save_email_data: {e}")
        return {"error": f"Failed to save data: {str(e)}"}
    

# Add an endpoint to retrieve CV text by ID
@app.get("/cv_text/{cv_id}")
def get_cv_text(cv_id: str):
    cv_doc = cvText_collection.find_one({'cv_id': cv_id}, {'_id': 0})
    if cv_doc:
        return cv_doc
    return {'error': 'CV text not found'}
    

# Add an endpoint to search CV text
@app.get("/search_cvs")
def search_cvs(query: str):
    # Search in extracted text and possible roles
    results = cvText_collection.find({
        '$or': [
            {'extracted_text': {'$regex': query, '$options': 'i'}},
            {'possible_roles': {'$regex': query, '$options': 'i'}}
        ]
    }, {'_id': 0})
    
    return list(results)

@app.get("/interns")
def get_all_interns():
    """Fetch all records from internsdata collection."""
    interns = list(collection.find({}))
    
    # Convert MongoDB objects to JSON serializable format
    for intern in interns:
        intern["_id"] = str(intern["_id"])  # Convert ObjectId to string
        intern["cv_id"] = str(intern.get("cv_id", ""))  # Ensure cv_id is a string

    return interns

def extract_pdf_from_email(service, message_id):
    """Extract PDF attachment from email"""
    try:
        message = service.users().messages().get(userId='me', id=message_id).execute()
        parts = message.get('payload', {}).get('parts', [])
        
        for part in parts:
            if part.get('filename', '').lower().endswith('.pdf'):
                if 'body' in part and 'attachmentId' in part['body']:
                    attachment = service.users().messages().attachments().get(
                        userId='me',
                        messageId=message_id,
                        id=part['body']['attachmentId']
                    ).execute()
                    
                    if attachment:
                        data = base64.urlsafe_b64decode(attachment['data'])
                        return {
                            'filename': part['filename'],
                            'data': data
                        }
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return None



@app.delete("/delete_all_records")
def delete_all_records():
    """Deletes all records from internsdata, cvs, and cvText collections."""
    interns_deleted = collection.delete_many({})
    cvs_deleted = cv_collection.delete_many({})
    cv_text_deleted = cvText_collection.delete_many({})
    
    return {
        "status": "success",
        "message": "All records deleted successfully",
        "interns_deleted": interns_deleted.deleted_count,
        "cvs_deleted": cvs_deleted.deleted_count,
        "cv_text_deleted": cv_text_deleted.deleted_count
    }


# New models for authentication
class UserCreate(BaseModel):
    fullName: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    fullName: str
    email: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Configure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configure JWT
SECRET_KEY = "23423423"  # Change this to a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")




# Helper functions for authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_email(email: str):
    return users_collection.find_one({"email": email})

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_collection.find_one({"_id": user_id})
    if user is None:
        raise credentials_exception
    return user

# Auth endpoints
@app.post("/signup", response_model=Token)
async def create_user(user: UserCreate):
    # Check if user already exists
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user.password)
    user_data = {
        "_id": user_id,
        "fullName": user.fullName,
        "email": user.email,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    
    users_collection.insert_one(user_data)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id}, expires_delta=access_token_expires
    )
    
    # Return token and user info
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "fullName": user.fullName,
            "email": user.email
        }
    }

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["_id"]}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["_id"],
            "fullName": user["fullName"],
            "email": user["email"]
        }
    }


# Request model
class QueryRequest(BaseModel):
    prompt: str
    filteredInterns: List[Dict]  # Expecting a list of dictionaries


# @app.post("/rank_cvs")
# def rank_cvs(request: QueryRequest):
#     # global data
#     # data = request.filteredInterns  # Store filtered interns in data

#     # Get the filtered interns from the request
#     filtered_interns = request.filteredInterns
    
#     # Check which of these interns still exist in the main collection
#     # This ensures we don't rank interns who have been shortlisted
#     valid_cv_ids = [intern["cv_id"] for intern in filtered_interns]
#     existing_interns = list(collection.find({"cv_id": {"$in": valid_cv_ids}}))
    
#     # Convert ObjectId to string for each intern
#     for intern in existing_interns:
#         intern["_id"] = str(intern["_id"])
    
#     # Update global data with only existing interns
#     global data
#     data = existing_interns

#     # Get today's date and the cutoff date (7 days from today)
#     today = datetime.today().date()
#     one_week_from_today = today + timedelta(days=7)

#     # Filter out CVs with starting_date that has already passed
#     valid_cvs = []
#     for cv in data:
#         try:
#             starting_date = datetime.strptime(cv["starting_date"], "%Y-%m-%d").date()
#             if today <= starting_date >= one_week_from_today:  # Keep CVs starting within the next 7 days
#                 valid_cvs.append(cv)
#         except (ValueError, KeyError, TypeError):
#             continue  # Skip if date is missing or invalid format

#     data = valid_cvs  # Update data with valid CVs

#     # Fetch extracted_text for each CV from the cvText collection
#     for cv in data:
#         cv_id = cv["cv_id"]
#         cv_text_doc = cvText_collection.find_one({"cv_id": cv_id}, {"_id": 0, "extracted_text": 1})
#         cv["extracted_text"] = cv_text_doc["extracted_text"] if cv_text_doc else None  # Add extracted text

#     custom_prompt = request.prompt.strip().lower()
#     summarized_texts = [cv["extracted_text"] for cv in data if cv["extracted_text"] is not None]

#     # Encode texts and compute similarity
#     cv_embeddings = encoder.encode(summarized_texts, convert_to_tensor=True)
#     prompt_embedding = encoder.encode(custom_prompt, convert_to_tensor=True)
#     cosine_scores = util.cos_sim(prompt_embedding, cv_embeddings)

#     # Rank CVs by similarity (highest score first)
#     ranked_indices = cosine_scores[0].argsort(descending=True)

#     # Create ranked list
#     ranked_cvs = []
#     for rank, i in enumerate(ranked_indices[:100], start=1): 
#         ranked_cvs.append({
#             "cvId": data[i]["cv_id"],
#             "name": data[i]["name"],
#             "education": data[i]["degree"],
#             "institute": data[i]["university"],
#             "email": data[i]["email"],  # This is correctly included here
#             "year": data[i]["current_year"],
#             "internshipPeriod": data[i]["internship_period"],
#             "workingMode": data[i]["working_mode"],
#             "possibleJobRoles": data[i]["expected_role"],
#             "rank": rank,
#             "cvLink": data[i]["cv_link"],
#             "contactNo": data[i]["contact_no"],
#             "skills": data[i]["skills"],
#             "startingDate": data[i]["starting_date"]
#         })

#     # ranked_cvs.sort(key=lambda x: x["startingDate"])

#     return {"ranked_cvs": ranked_cvs}


@app.post("/rank_cvs")
def rank_cvs(request: QueryRequest):
    # Get the filtered interns from the request
    filtered_interns = request.filteredInterns
    
    # Check which of these interns still exist in the main collection
    # This ensures we don't rank interns who have been shortlisted
    valid_cv_ids = [intern["cv_id"] for intern in filtered_interns if "cv_id" in intern]
    
    if not valid_cv_ids:
        return {"ranked_cvs": [], "message": "No valid CVs to rank"}
    
    existing_interns = list(collection.find({"cv_id": {"$in": valid_cv_ids}}))
    
    # Convert ObjectId to string for each intern
    for intern in existing_interns:
        intern["_id"] = str(intern["_id"])
    
    # Update global data with only existing interns
    global data
    data = existing_interns
    
    if not data:
        return {"ranked_cvs": [], "message": "No matching interns found in database"}

    # Get today's date and the cutoff date (7 days from today)
    today = datetime.today().date()
    one_week_from_today = today + timedelta(days=7)

    # Filter out CVs with starting_date that has already passed
    valid_cvs = []
    for cv in data:
        try:
            if "starting_date" in cv and cv["starting_date"]:
                starting_date = datetime.strptime(cv["starting_date"], "%Y-%m-%d").date()
                # Fixed the comparison logic
                if today <= starting_date >= one_week_from_today:  # Keep all CVs with starting date today or in future
                    valid_cvs.append(cv)
            else:
                valid_cvs.append(cv)  # Keep CVs without starting date
        except (ValueError, KeyError, TypeError):
            valid_cvs.append(cv)  # Keep CVs with invalid date format

    data = valid_cvs  # Update data with valid CVs
    
    if not data:
        return {"ranked_cvs": [], "message": "No Interns Found."}

    # Fetch extracted_text for each CV from the cvText collection
    cvs_with_text = []
    for cv in data:
        if "cv_id" in cv:
            cv_id = cv["cv_id"]
            cv_text_doc = cvText_collection.find_one({"cv_id": cv_id}, {"_id": 0, "extracted_text": 1})
            if cv_text_doc and cv_text_doc.get("extracted_text"):
                cv["extracted_text"] = cv_text_doc["extracted_text"]
                cvs_with_text.append(cv)
            else:
                # No text found for this CV
                print(f"No extracted text found for CV ID: {cv_id}")
    
    data = cvs_with_text  # Only keep CVs with extracted text
    
    if not data:
        return {"ranked_cvs": [], "message": "No CVs with extracted text found"}

    custom_prompt = request.prompt.strip().lower()
    summarized_texts = [cv["extracted_text"] for cv in data if cv.get("extracted_text")]
    
    if not summarized_texts:
        return {"ranked_cvs": [], "message": "No CV texts to analyze"}

    # Encode texts and compute similarity
    cv_embeddings = encoder.encode(summarized_texts, convert_to_tensor=True)
    prompt_embedding = encoder.encode(custom_prompt, convert_to_tensor=True)
    cosine_scores = util.cos_sim(prompt_embedding, cv_embeddings)

    # Rank CVs by similarity (highest score first)
    ranked_indices = cosine_scores[0].argsort(descending=True)

    # Create ranked list
    ranked_cvs = []
    for rank, i in enumerate(ranked_indices, start=1): 
        if i < len(data):  # Ensure index is valid
            cv = data[i]
            ranked_cvs.append({
                "cvId": cv["cv_id"],
                "name": cv.get("name", "Unknown"),
                "education": cv.get("degree", ""),
                "institute": cv.get("university", ""),
                "email": cv.get("email", ""),  # Ensure email is included
                "year": cv.get("current_year", ""),
                "internshipPeriod": cv.get("internship_period", ""),
                "workingMode": cv.get("working_mode", ""),
                "possibleJobRoles": cv.get("expected_role", []),
                "rank": rank,
                "cvLink": cv.get("cv_link", ""),
                "contactNo": cv.get("contact_no", ""),
                "skills": cv.get("skills", {}),
                "startingDate": cv.get("starting_date", "")
            })

    # ranked_cvs.sort(key=lambda x: x["startingDate"])
    print(f"Ranked {len(ranked_cvs)} CVs successfully")
    return {"ranked_cvs": ranked_cvs}

@app.post("/shortlist/{cv_id}")
async def shortlist_intern(cv_id: str):
    global data
    intern = collection.find_one({"cv_id": cv_id})

    if not intern:
        raise HTTPException(status_code=404, detail="Intern not found")

    # Insert into shortlist collection
    shortlist_collection.insert_one(intern)

    # Remove from internsdata collection
    collection.delete_one({"cv_id": cv_id})

    # Remove from filteredInterns (modify in place)
    data[:] = [cv for cv in data if cv["cv_id"] != cv_id]

    return {"message": "Intern shortlisted successfully"}

@app.delete("/remove/{cv_id}")
async def remove_intern(cv_id: str):
    global data
    result = collection.delete_one({"cv_id": cv_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Intern not found")

    # Remove from filteredInterns (modify in place)
    data[:] = [cv for cv in data if cv["cv_id"] != cv_id]

    return {"message": "Intern removed successfully"}

@app.get("/shortlisted-interns")
def get_shortlisted_interns():
    """Fetch all shortlisted interns from the shortlist collection."""
    shortlisted = list(shortlist_collection.find({}))
        
    # Convert MongoDB objects to JSON serializable format
    for intern in shortlisted:
        intern["_id"] = str(intern["_id"])  # Convert ObjectId to string
        intern["cv_id"] = str(intern.get("cv_id", ""))  # Ensure cv_id is a string

    return {"shortlisted_interns": shortlisted}

@app.delete("/clear-shortlisted")
def clear_shortlisted_interns():
    """Delete all records from the shortlist collection."""
    result = shortlist_collection.delete_many({})
    
    return {
        "status": "success",
        "message": "All shortlisted interns removed successfully",
        "deleted_count": result.deleted_count
    }

@app.delete("/remove-shortlisted/{cv_id}")
async def remove_shortlisted_intern(cv_id: str):
    """Remove a specific intern from the shortlist collection."""
    result = shortlist_collection.delete_one({"cv_id": cv_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Shortlisted intern not found")

    return {"message": "Intern removed from shortlist successfully"}

#----------------------------------- CLEAR OLD INTERNS----------------------------------------------
def _parse_iso_date(s: str) -> datetime:
    """
    Parse common ISO-like dates. Treat date-only as midnight UTC.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")

    # date-only
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.fromisoformat(s)  # naive
        return dt.replace(tzinfo=timezone.utc)

    # datetime ISO
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _coerce_starting_date(doc) -> datetime | None:
    v = doc.get("startingDate")
    if v is None:
        return None
    if isinstance(v, datetime):
        # normalize to UTC aware
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
    if isinstance(v, str):
        try:
            return _parse_iso_date(v)
        except Exception:
            return None
    return None


@app.delete("/interns/clear-old")
def clear_old_interns():
    # You should inject this collection from your app (shown below)
    global collection  # type: ignore

    if collection is None:
        raise HTTPException(status_code=500, detail="DB not initialized")

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - relativedelta(months=3)

    # If startingDate is stored as a BSON Date (datetime) in MongoDB:
    # This is fast and uses Mongo filtering.
    mongo_result = collection.delete_many({"startingDate": {"$lte": cutoff}})
    deleted_by_date_type = mongo_result.deleted_count

    # If you ALSO have older records where startingDate is stored as string,
    # Mongo can't reliably compare strings to datetimes, so we handle those.
    # We find docs with string startingDate and delete by computed list.
    string_docs = collection.find(
        {"startingDate": {"$type": "string"}},
        {"_id": 1, "startingDate": 1}
    )

    ids_to_delete = []
    for d in string_docs:
        dt = _coerce_starting_date(d)
        if dt and dt <= cutoff:
            ids_to_delete.append(d["_id"])

    deleted_string = 0
    if ids_to_delete:
        r2 = collection.delete_many({"_id": {"$in": ids_to_delete}})
        deleted_string = r2.deleted_count

    return {
        "deleted_count": deleted_by_date_type + deleted_string,
        "cutoff_date": cutoff.date().isoformat(),
        "now_utc": now_utc.isoformat(),
    }

# @app.post("/hire-intern/{cv_id}")
# async def hire_intern(cv_id: str):
#     """Mark an intern as hired, send them an email notification, and remove them from the shortlist."""
#     intern = shortlist_collection.find_one({"cv_id": cv_id})
    
#     if not intern:
#         raise HTTPException(status_code=404, detail="Intern not found")
    
#     # Get the intern's email
#     intern_email = intern.get("email")
#     if not intern_email:
#         raise HTTPException(status_code=400, detail="Intern email not found")
    
#     # Updated subject line
#     subject = "Internship at SLT Digital Lab"
    
#     # Updated email body with exact spacing preserved
#     body = """Dear Candidate,


# Please be informed that you have been selected for the internship at SLT Digital Lab unit. You are required to obtain a police report and submit a scanned copy before 30 days from this notification via,


# https://forms.office.com/r/VJYrmWQ9i1


# Please note that a first come first serve basis will be considered for internship allocation in SLT Digital Lab.


# Refer the FAQ given below. If you have any query, you can send via,


# https://forms.office.com/r/NmDAdr7ZmW


# Additionally, the below documents shall be ready as those will be requested once your police report is verified,

# • NIC Copy

# • Faculty Letter

# • Copy of Vaccination report with booster


# Regards,

# Internship Management System Digital Lab


# FAQ:

# 1. Is there an interview?


# No, You have already been selected for the internship. After signing the agreements, you can start the internship. However the internship allocation will be first come first serve basis and the selection will be invalid if you could n't turned up before the next candidate.


# 2. Is the police report and faculty letter compulsory?


# Yes, without the original police report and the faculty letter, you can't sign the agreement.


# 3. Can we sign the agreement except on Tuesdays?


# No, you can sign the agreements only on Tuesdays.


# 4. To whom do we need to address the police report and the faculty letter?


# HR Department,

# Sri Lanka Telecom HQ building,

# Lotus Road,

# Colombo 01


# 5. Is the internship remote or work-from-home?


# It is hybrid mode. If your institution/university requires you to work in an office, we arrange that too. Apart from that, you have to report to the office when your supervisor request based on work arrangements.


# 6. Are we getting any salary?


# No, it's an unpaid internship.


# 7. What's the internship duration?


# According to your institution/university's requirements, we can arrange the internship.


# 8. Can we participate in the lectures during office hours?


# No, without prior approval from your supervisor. Such requests should be made with institution/university's original request/guideline."""
    
#     try:
#         # Setup the email
#         message = MIMEMultipart()
#         message["From"] = EMAIL_SENDER
#         message["To"] = intern_email
#         message["Subject"] = subject
        
#         # Attach the body to the email
#         message.attach(MIMEText(body, "plain"))
        
#         # Connect to Gmail's SMTP server with proper error handling
#         try:
#             with smtplib.SMTP("smtp.gmail.com", 587) as server:
#                 server.ehlo()  # Can help with connection issues
#                 server.starttls()  # Secure the connection
#                 server.ehlo()  # Some servers need this twice
#                 server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#                 server.send_message(message)
#                 print(f"Email sent successfully to {intern_email}")
                
#                 # Now that the email was sent successfully, remove the intern from the shortlist
#                 shortlist_collection.delete_one({"cv_id": cv_id})
#                 print(f"Intern {intern['name']} removed from shortlist after hiring")
                
#         except smtplib.SMTPAuthenticationError as auth_error:
#             print(f"SMTP Authentication Error: {auth_error}")
#             raise HTTPException(
#                 status_code=500, 
#                 detail="Email authentication failed. Please check email credentials."
#             )
#         except smtplib.SMTPException as smtp_error:
#             print(f"SMTP Error: {smtp_error}")
#             raise HTTPException(
#                 status_code=500, 
#                 detail=f"SMTP error occurred: {str(smtp_error)}"
#             )
        
#         # If we got here, both the email was sent and the intern was removed from shortlist
#         return {"message": f"Hiring email sent to {intern['name']} at {intern_email} and removed from shortlist"}
        
#     except Exception as e:
#         print(f"Error in hire process: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to complete hiring process: {str(e)}")

# Create a model for the hire request that includes deadline_date
class HireRequest(BaseModel):
    deadline_date: str

# @app.post("/hire-intern/{cv_id}")
# async def hire_intern(cv_id: str, request: HireRequest):
#     """Mark an intern as hired, send them an email notification with deadline date, and remove them from the shortlist."""
#     intern = shortlist_collection.find_one({"cv_id": cv_id})
    
#     if not intern:
#         raise HTTPException(status_code=404, detail="Intern not found")
    
#     # Get the intern's email
#     intern_email = intern.get("email")
#     if not intern_email:
#         raise HTTPException(status_code=400, detail="Intern email not found")
    
#     # Process the deadline date
#     deadline_date = request.deadline_date
    
#     # Format the date for display if needed (optional)
#     try:
#         # Parse the date to ensure it's valid
#         date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
#         # Format it in a more readable format if desired
#         formatted_date = date_obj.strftime("%B %d, %Y")  # e.g., "May 15, 2025"
#     except ValueError:
#         # If date parsing fails, just use the original string
#         formatted_date = deadline_date
    
#     # Updated subject line
#     subject = "Internship at SLT Digital Lab"
    
#     # Updated email body with deadline date
#     body = f"""Dear Candidate,


# Please be informed that you have been selected for the internship at SLT Digital Lab unit. You are required to obtain a police report and submit a scanned copy before {formatted_date} via,


# https://forms.office.com/r/VJYrmWQ9i1


# Please note that a first come first serve basis will be considered for internship allocation in SLT Digital Lab.


# Refer the FAQ given below. If you have any query, you can send via,


# https://forms.office.com/r/NmDAdr7ZmW


# Additionally, the below documents shall be ready as those will be requested once your police report is verified,

# • NIC Copy

# • Faculty Letter

# • Copy of Vaccination report with booster


# Regards,

# Internship Management System Digital Lab


# FAQ:

# 1. Is there an interview?


# No, You have already been selected for the internship. After signing the agreements, you can start the internship. However the internship allocation will be first come first serve basis and the selection will be invalid if you could n't turned up before the next candidate.


# 2. Is the police report and faculty letter compulsory?


# Yes, without the original police report and the faculty letter, you can't sign the agreement.


# 3. Can we sign the agreement except on Tuesdays?


# No, you can sign the agreements only on Tuesdays.


# 4. To whom do we need to address the police report and the faculty letter?


# HR Department,

# Sri Lanka Telecom HQ building,

# Lotus Road,

# Colombo 01


# 5. Is the internship remote or work-from-home?


# It is hybrid mode. If your institution/university requires you to work in an office, we arrange that too. Apart from that, you have to report to the office when your supervisor request based on work arrangements.


# 6. Are we getting any salary?


# No, it's an unpaid internship.


# 7. What's the internship duration?


# According to your institution/university's requirements, we can arrange the internship.


# 8. Can we participate in the lectures during office hours?


# No, without prior approval from your supervisor. Such requests should be made with institution/university's original request/guideline."""
    
#     try:
#         # Setup the email
#         message = MIMEMultipart()
#         message["From"] = EMAIL_SENDER
#         message["To"] = intern_email
#         message["Subject"] = subject
        
#         # Attach the body to the email
#         message.attach(MIMEText(body, "plain"))
        
#         # Connect to Gmail's SMTP server with proper error handling
#         try:
#             with smtplib.SMTP("smtp.gmail.com", 587) as server:
#                 server.ehlo()  # Can help with connection issues
#                 server.starttls()  # Secure the connection
#                 server.ehlo()  # Some servers need this twice
#                 server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#                 server.send_message(message)
#                 print(f"Email sent successfully to {intern_email} with deadline date: {formatted_date}")
                
#                 # Now that the email was sent successfully, remove the intern from the shortlist
#                 shortlist_collection.delete_one({"cv_id": cv_id})
#                 print(f"Intern {intern['name']} removed from shortlist after hiring")
                
#         except smtplib.SMTPAuthenticationError as auth_error:
#             print(f"SMTP Authentication Error: {auth_error}")
#             raise HTTPException(
#                 status_code=500, 
#                 detail="Email authentication failed. Please check email credentials."
#             )
#         except smtplib.SMTPException as smtp_error:
#             print(f"SMTP Error: {smtp_error}")
#             raise HTTPException(
#                 status_code=500, 
#                 detail=f"SMTP error occurred: {str(smtp_error)}"
#             )
        
#         # If we got here, both the email was sent and the intern was removed from shortlist
#         return {"message": f"Hiring email sent to {intern['name']} at {intern_email} with deadline date: {formatted_date}"}
        
#     except Exception as e:
#         print(f"Error in hire process: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to complete hiring process: {str(e)}")



# Model for accepting an intern
class AcceptInternRequest(BaseModel):
    start_date: str
    
# Model for marking an intern as verified
class VerifyInternRequest(BaseModel):
    is_verified: bool = True

@app.post("/hire-intern/{cv_id}")
async def hire_intern(
    cv_id: str, 
    deadline_date: str = Form(...),
    email_subject: str = Form(...),
    email_body: str = Form(...),
    attachment: UploadFile = File(None),
    use_default_attachment: bool = Form(True)  # Default to using the attachment
):
    """
    Mark an intern as hired, send them a customized email notification, and remove them from the shortlist.
    """
    intern = shortlist_collection.find_one({"cv_id": cv_id})
    
    if not intern:
        raise HTTPException(status_code=404, detail="Intern not found")
    
    # Get the intern's email
    intern_email = intern.get("email")
    if not intern_email:
        raise HTTPException(status_code=400, detail="Intern email not found")
    
    # Process the deadline date
    try:
        # Parse the date to ensure it's valid
        date_obj = datetime.strptime(deadline_date, "%Y-%m-%d")
        # Format it in a more readable format
        formatted_date = date_obj.strftime("%B %d, %Y")  # e.g., "May 15, 2025"
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    try:
        # Setup the email
        message = MIMEMultipart()
        message["From"] = EMAIL_SENDER
        message["To"] = intern_email
        message["Subject"] = email_subject
        
        # Convert Markdown-style formatting to HTML
        html_body = email_body.replace("**", "<strong>", 1)
        i = 1
        while "**" in html_body:
            if i % 2 == 0:
                html_body = html_body.replace("**", "</strong>", 1)
            else:
                html_body = html_body.replace("**", "<strong>", 1)
            i += 1
            
        # Replace line breaks with HTML breaks
        html_body = html_body.replace('\n', '<br>')
        
        # Create a proper HTML document
        full_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.5; color: #333; }}
                strong {{ font-weight: bold; }}
                .footer {{ margin-top: 20px; color: #666; font-size: 0.9em; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        
        # Attach both plain text and HTML versions (for email clients that don't support HTML)
        part1 = MIMEText(email_body, 'plain')
        part2 = MIMEText(full_html, 'html')
        message.attach(part1)
        message.attach(part2)
        
        # Handle attachment
        attachment_path = None
        temp_file_created = False
        
        if attachment and attachment.filename:
            # Save the uploaded file temporarily
            temp_attachment_path = f"{ATTACHMENTS_DIR}/temp_{attachment.filename}"
            with open(temp_attachment_path, "wb") as temp_file:
                content = await attachment.read()
                temp_file.write(content)
            attachment_path = temp_attachment_path
            temp_file_created = True
        elif use_default_attachment and os.path.exists(DEFAULT_ATTACHMENT_PATH):
            attachment_path = DEFAULT_ATTACHMENT_PATH
        
        # Attach the file if available
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as attach_file:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attach_file.read())
            
            # Encode and add header
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            if filename.startswith("temp_") and temp_file_created:
                # Use the original filename without the temp_ prefix
                filename = attachment.filename
                
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={filename}",
            )
            message.attach(part)
            
            # Delete temporary file if it was created
            if temp_file_created:
                try:
                    os.remove(attachment_path)
                    print(f"Temporary file deleted: {attachment_path}")
                except Exception as e:
                    print(f"Error deleting temporary file: {e}")
        
        # Connect to Gmail's SMTP server with proper error handling
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()  # Can help with connection issues
                server.starttls()  # Secure the connection
                server.ehlo()  # Some servers need this twice
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(message)
                print(f"Email sent successfully to {intern_email} with deadline date: {formatted_date}")
                
                # Add the hired intern to the hired_interns collection with additional fields
                hired_intern = {**intern}
                hired_intern["document_deadline"] = deadline_date
                hired_intern["is_verified"] = False
                hired_intern["is_accepted"] = False
                hired_intern["hire_date"] = datetime.now().isoformat()
                hired_intern["start_date"] = None
                hired_intern["end_date"] = None
                hired_intern["email_subject"] = email_subject
                hired_intern["email_body"] = email_body
                
                # Insert into hired_interns collection
                hired_interns_collection.insert_one(hired_intern)
                
                # Remove from shortlist collection
                shortlist_collection.delete_one({"cv_id": cv_id})
                print(f"Intern {intern['name']} removed from shortlist and added to hired interns")
                
        except smtplib.SMTPAuthenticationError as auth_error:
            print(f"SMTP Authentication Error: {auth_error}")
            raise HTTPException(
                status_code=500, 
                detail="Email authentication failed. Please check email credentials."
            )
        except smtplib.SMTPException as smtp_error:
            print(f"SMTP Error: {smtp_error}")
            raise HTTPException(
                status_code=500, 
                detail=f"SMTP error occurred: {str(smtp_error)}"
            )
        
        # If we got here, both the email was sent and the intern was processed
        return {"message": f"Hiring email sent to {intern['name']} at {intern_email} with deadline date: {formatted_date}"}
        
    except Exception as e:
        print(f"Error in hire process: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete hiring process: {str(e)}")

# Endpoint to get all hired interns
@app.get("/hired-interns")
def get_hired_interns(name: Optional[str] = None, date: Optional[str] = None, status: Optional[str] = None):
    """Fetch hired interns with optional filters."""
    query = {}
    
    # Add name filter if provided
    if name:
        query["name"] = {"$regex": name, "$options": "i"}
    
    # Add date filter if provided
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            
            if status == "pending":
                # For pending interns, filter by hire_date
                date_start = date_obj.replace(hour=0, minute=0, second=0)
                date_end = date_obj.replace(hour=23, minute=59, second=59)
                query["hire_date"] = {
                    "$gte": date_start.isoformat(),
                    "$lte": date_end.isoformat()
                }
            elif status == "active":
                # For active interns, filter by start_date
                query["start_date"] = date
        except ValueError:
            # Invalid date format, ignore this filter
            pass
    
    # Add verification/acceptance status filters
    if status == "pending":
        # Pending interns are either not verified or verified but not accepted
        query["$or"] = [
            {"is_verified": False},
            {"is_verified": True, "is_accepted": False}
        ]
    elif status == "active":
        # Active interns are both verified and accepted
        query["is_verified"] = True
        query["is_accepted"] = True
    
    # Execute the query
    interns = list(hired_interns_collection.find(query))
    
    # Convert MongoDB objects to JSON serializable format
    for intern in interns:
        intern["_id"] = str(intern["_id"])  # Convert ObjectId to string
        intern["cv_id"] = str(intern.get("cv_id", ""))  # Ensure cv_id is a string
    
    return {"hired_interns": interns}

# Endpoint to mark an intern as verified
@app.post("/hired-interns/{cv_id}/verify")
async def verify_intern(cv_id: str, request: VerifyInternRequest):
    """Mark an intern as verified (documents received and verified)."""
    # Find and update the intern
    result = hired_interns_collection.update_one(
        {"cv_id": cv_id},
        {"$set": {"is_verified": request.is_verified}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Intern not found")
    
    return {"message": "Intern verified successfully"}

# Endpoint to accept an intern and set start/end dates
@app.post("/hired-interns/{cv_id}/accept")
async def accept_intern(cv_id: str, request: AcceptInternRequest):
    """Accept an intern and set their start and end dates."""
    # Find the intern first to get their internship period
    intern = hired_interns_collection.find_one({"cv_id": cv_id})
    
    if not intern:
        raise HTTPException(status_code=404, detail="Intern not found")
    
    # Check if the intern is already verified
    if not intern.get("is_verified", False):
        raise HTTPException(status_code=400, detail="Intern must be verified before acceptance")
    
    # Parse the start date
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start date format. Use YYYY-MM-DD")
    
    # Calculate end date based on internship period
    internship_period = intern.get("internship_period", "")
    
    # Extract numeric value from internship period (assuming format like "3 months")
    try:
        months = int(internship_period.split()[0])
    except (ValueError, IndexError, AttributeError):
        # Default to 3 months if parsing fails
        months = 3
    
    # Calculate end date
    end_date = start_date + relativedelta(months=months)
    
    # Update the intern record
    result = hired_interns_collection.update_one(
        {"cv_id": cv_id},
        {"$set": {
            "is_accepted": True,
            "start_date": request.start_date,
            "end_date": end_date.strftime("%Y-%m-%d")
        }}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Intern not found")
    
    return {
        "message": "Intern accepted successfully", 
        "start_date": request.start_date,
        "end_date": end_date.strftime("%Y-%m-%d")
    }