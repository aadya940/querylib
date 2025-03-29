from querylib.extractor import DocstringExtractor
from querylib.rag import DocumentationRAG
import os

import pprint 

# Extractor
extractor = DocstringExtractor("numpy.fft", max_workers=os.cpu_count())
docs = extractor.extract_docs()
extractor.save_to_json(docs, "numpy.json")

# RAG
rag_system = DocumentationRAG("numpy.json", similarity_top_k=16)
response = rag_system.ask_query("How to perform fast fourier transform on a signal using Numpy?")

for res in response["results"]:
    pprint.pprint(res)
