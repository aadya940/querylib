import pytest
import json
from querylib.rag import DocumentationRAG
from querylib.extractor import DocstringExtractor

@pytest.fixture
def sample_json_file(tmp_path):
    # Sample JSON data for testing with 'path' included
    d = DocstringExtractor("sklearn", max_workers=4)
    doc = d.extract_docs()
    # Create a temporary JSON file
    json_file = tmp_path / "test_sample.json"
    d.save_to_json(doc, filename=str(json_file))
    return json_file

def test_ask_query(sample_json_file):
    rag = DocumentationRAG(str(sample_json_file))
    response = rag.ask_query("How to create a Linear Regression Model?")
    
    # Check the structure of the response
    assert isinstance(response, dict)
    assert "query" in response
    assert "results" in response
    
    # Check if the query is correctly returned
    assert response["query"] == "How to create a Linear Regression Model?"
    
    # Check if results is a list
    assert isinstance(response["results"], list)
    
    # Optionally check if metadata is returned
    assert "metadata" in response["results"][0]
    assert "score" in response["results"][0]
    
    print(response["result"][0])
