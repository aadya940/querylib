from querylib import DocumentationRAG
import pprint

r = DocumentationRAG("sklearn.json", similarity_top_k=8)
res = r.ask_query("How to perform Simple Linear Regression using sklearn?")
for i in res["results"]:
    pprint.pprint(i)
