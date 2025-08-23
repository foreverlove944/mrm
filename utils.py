import chromadb
from chromadb.utils import embedding_functions


def get_collection(chroma_path,ef,name="musique_collection"):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=name,embedding_function=ef)
    print(f"search in {name}")
    return collection
