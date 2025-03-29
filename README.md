# QueryLib

QueryLib is a Python library designed for extracting documentation from Python packages and implementing a Retrieval-Augmented Generation (RAG) system for querying that documentation.

## Features

- **Docstring Extraction**: Efficiently extracts documentation from Python modules.
- **RAG System**: Optimized system for querying documentation using embeddings and FAISS for fast retrieval.

## Installation

To install the required dependencies, you can use pip:

```bash
git clone ...
cd querylib/
pip install -r requirements.txt
pip install .
```

## Usage

### DocstringExtractor

To extract documentation from a Python package:

```python
from querylib.extractor import DocstringExtractor

extractor = DocstringExtractor("numpy")
docs = extractor.extract_docs()
extractor.save_to_json(docs, "numpy.json")
```

### DocumentationRAG

To set up the RAG system:

```python
from querylib.rag import DocumentationRAG

rag_system = DocumentationRAG("numpy.json")
response = rag_system.ask_query("How to add to vectors together using Numpy?")
print(response)
```

## Contributing

Feel free to submit issues or pull requests to improve the library.

## License

This project is licensed under the MIT License.
