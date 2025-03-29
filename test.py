from querylib.extractor import DocstringExtractor
from querylib.rag import DocumentationRAG

# Extractor
extractor = DocstringExtractor("numpy")
docs = extractor.extract_docs()
extractor.save_to_json(docs, "numpy.json")

# RAG
rag_system = DocumentationRAG("numpy.json")
response = rag_system.ask_query("How to add two vectors together using Numpy?")
print(response)
