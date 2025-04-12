import io
import os
import datetime
import gridfs
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

def upload_marks_file_to_gridfs(file, user, subject: str):
    """
    Uploads a marks file to GridFS with custom metadata.

    Only users with a role of "Professor" or "HOD" are allowed to perform this operation.

    Parameters:
        file: The uploaded file object (e.g., from st.file_uploader).
        user: A dictionary containing the current user's details (including username, role, department).
        subject: A string representing the subject associated with the marks file.

    Returns:
        file_id: The GridFS file ID if the upload is successful.
    """
    # Enforce that only Professors or HODs can upload marks files.
    if user.get("role") not in ["Professor", "HOD"]:
        raise PermissionError("User is not authorized to upload marks files.")
    
    # Use GridFS with a custom collection name "marks" so that files are stored in marks.files and marks.chunks.
    fs = gridfs.GridFS(db, collection="marks")
    
    # In your upload function, after retrieving the subject input:
    subject_input = subject.strip()  # e.g., "Thermodynamics, Machine Design"
    subjects = [sub.strip() for sub in subject_input.split(",")]
    
    # Prepare metadata for the file, including subject and department details.
    metadata = {
        "department": user.get("department"),
        "uploaded_by": user.get("username"),
        "subject": subject,
        "upload_date": datetime.datetime.utcnow().isoformat()
    }
    
    # Upload the file's binary content using file.getvalue() (suitable for Streamlit's UploadedFile object).
    file_id = fs.put(file.getvalue(), filename=file.name, metadata=metadata)
    
    return file_id

def get_file_as_file_object_from_gridfs(file_id, db, original_filename=None):
    """
    Retrieves a file from GridFS using its file_id and returns a BytesIO file-like object.
    Sets the file name (with extension) so that downstream processing knows the file type.
    """
    fs = gridfs.GridFS(db, collection="marks")  # Using custom bucket 'marks'
    grid_file = fs.get(file_id)
    file_data = grid_file.read()
    file_obj = io.BytesIO(file_data)
    # If an original filename is provided, use it; otherwise, use the filename from GridFS.
    if original_filename:
        file_obj.name = original_filename
    else:
        file_obj.name = grid_file.filename
    return file_obj

def get_latest_marks_file_for_department(department: str, db):
    """
    Retrieves the most recent marks file uploaded for a given department from GridFS.
    
    Parameters:
       department: The department name (e.g., "ME")
       db: The MongoDB database object
       
    Returns:
       A tuple (file_obj, file_doc), where file_obj is a BytesIO file-like object 
       (with a valid file name) and file_doc contains the file's metadata.
       Returns (None, None) if no matching file is found.
    """
    file_doc = db["marks.files"].find_one(
        {"metadata.department": department},
        sort=[("metadata.upload_date", -1)]
    )
    if file_doc is None:
        return None, None
    file_id = file_doc["_id"]
    file_obj = get_file_as_file_object_from_gridfs(file_id, db, original_filename=file_doc.get("filename"))
    return file_obj, file_doc
