# # data_manager/data_manager.py
# from pymongo import MongoClient
# from typing import Optional, List, Tuple

# class DataManager:
#     """
#     Provides an interface to the MongoDB institutional database.
#     """
#     def __init__(self, uri: str, db_name: str):
#         self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
#         self.db = self.client[db_name]
#         # Test connection
#         self.db.list_collection_names()

#     def fetch_all_records(self, collections: Optional[List[str]] = None) -> List[Tuple[str, dict]]:
#         """
#         Return all records from the specified collections.
#         """
#         if collections is None:
#             collections = ["students", "professors", "departments"]
#         records = []
#         for coll_name in collections:
#             if coll_name in self.db.list_collection_names():
#                 collection = self.db[coll_name]
#                 for doc in collection.find({}):
#                     if "_id" in doc:
#                         doc["_id"] = str(doc["_id"])
#                     records.append((coll_name, doc))
#         return records

#     def count_documents(self, collection_name: str, filter_query: Optional[dict] = None) -> int:
#         """
#         Count documents in a collection that match the filter_query.
#         """
#         if filter_query is None:
#             filter_query = {}
#         return self.db[collection_name].count_documents(filter_query)

#     def find_documents(self, collection_name: str, filter_query: Optional[dict] = None, projection: Optional[dict] = None) -> List[dict]:
#         """
#         Find documents in a collection matching the filter_query.
#         """
#         if filter_query is None:
#             filter_query = {}
#         collection = self.db[collection_name]
#         cursor = collection.find(filter_query, projection)
#         docs = list(cursor)
#         for doc in docs:
#             if "_id" in doc:
#                 doc["_id"] = str(doc["_id"])
#         return docs
