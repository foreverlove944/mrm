import argparse
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompt.rag import rag_prompt
from utils import get_collection
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./standard_question_answer/musique_examples.jsonl')
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
model = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model="gpt-3.5-turbo",
            temperature=0,
            
        )
collection = get_collection(chroma_path=chroma_path,ef=ef)
chain = (
            {
                "question": lambda x: x["question"],
                "documents": lambda x: x["documents"]
            }
            | rag_prompt
            | model
            | StrOutputParser()
        )

def search_documents(query):
        result = collection.query(
            query_texts=[query],
            n_results=10,
        )
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        return documents,metadatas



basename = os.path.basename(file_path)
basename = "rag_" + basename
f_answer = open(os.path.join("./result",basename),"w",encoding="utf-8")
question = []
with open(file_path,"r",encoding='utf-8') as f:
    for i in f.readlines():
        i_dict = json.loads(i)
        question.append(i_dict)
for i in tqdm(question):
    question = i["question"]
    documents,metadatas = search_documents(question)
    answer = chain.invoke({"question":i["question"],"documents":documents})
    i["model_answer"] = answer
    i["documents"] = documents
    i["metadatas"] = metadatas
    json.dump(i,f_answer,ensure_ascii=False)
    f_answer.write("\n")