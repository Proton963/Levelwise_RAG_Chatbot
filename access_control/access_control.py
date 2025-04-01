import os
import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables and set up MongoDB connection
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME", "collegeDB")

if not mongo_uri:
    raise ValueError("MONGO_URI environment variable not set.")
    
client = MongoClient(mongo_uri)
db = client[mongo_db_name]

def store_uploaded_file(file, user):
    """
    Saves an uploaded file to disk and stores its metadata in the 'user_documents' collection.
    
    Parameters:
      file: The uploaded file object (e.g., from st.file_uploader)
      user: A dictionary containing the current user's details (e.g., username, role)
    
    Returns:
      file_metadata: A dictionary containing details about the uploaded file.
    """
    # Ensure the uploads directory exists
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the file locally
    file_path = os.path.join(upload_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Create metadata for the file
    file_metadata = {
        "filename": file.name,
        "filepath": file_path,
        "uploaded_at": datetime.datetime.utcnow().isoformat(),
        "uploaded_by": user.get("username"),
        "role": user.get("role")
    }
    
    # Insert or update the user's document with this file metadata in 'user_documents'
    db.user_documents.update_one(
        {"username": user.get("username")},
        {"$push": {"documents": file_metadata}},
        upsert=True
    )
    
    return file_metadata
