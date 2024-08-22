
import argparse
import json
import os

import chromadb
from chromadb.utils import embedding_functions

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./dataset/musique_examples.jsonl')
parser.add_argument('--api_key', type=str,help='openai api key')
parser.add_argument('--chroma_path', default="./chroma_db", help='chroma db path')
parser.add_argument('--api_base', default=None)
args = parser.parse_args()
api_key = args.api_key
file_path = args.input_file
chroma_path = args.chroma_path
api_base = args.api_base


ef  =  embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name= "text-embedding-large",  #"text-embedding-large",
            api_base=api_base ,   #If you are using another service provider, please add the `api_base` parameter.

        )   



def create_directory(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"{path} has been establish")
        else:
            print(f"{path} has been establish")
    except OSError as error:
        print(f"{path} error: {error}")

def vectorize_documents(file_path, chroma_path):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.create_collection(name="musique_collection",embedding_function=ef)
    question_list = []
    with open(file_path, 'r') as f:
        for i in f.readlines():
            i_dict = json.loads(i)
            question_list.append(i_dict)
    documents_list = []
    metadata_list =[]
    for i in question_list:
        documents = i["paragraphs"]
        id = i["id"]
        for j in documents:
            title = j["title"]
            paragraph_text = j["paragraph_text"]
            paragraph = title + "\n" + paragraph_text
            metadata = j["is_supporting"]
            documents_list.append(paragraph)
            metadata_list.append({"id":id,"is_supporting":metadata})
    documents_num = len(documents_list)
    batch_size = 20
    for i in range(0, documents_num, batch_size):
        end = min(i+batch_size, documents_num)
        batch_documents = documents_list[i:end]
        batch_metadata = metadata_list[i:end]
        collection.add(
            documents=batch_documents,metadatas=batch_metadata,
            ids=[str(i) for i in range(i, end)])
        print(f"musique_collection:{collection.count()}")
    print("Vectorization completed")
            
if __name__ == "__main__":
    create_directory(chroma_path)
    vectorize_documents(file_path, chroma_path)
