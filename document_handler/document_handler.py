# import os
# from langchain_community.document_loaders import PyPDFLoader

# def load_documents(uploaded_files, upload_dir):
#     documents = []
#     if not os.path.exists(upload_dir):
#         os.makedirs(upload_dir)
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(upload_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         loader = PyPDFLoader(file_path)
#         documents.extend(loader.load())
#     return documents
