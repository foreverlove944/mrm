import argparse
import json
import os

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from prompt.mrm_prompt import (mrm_prompt_design_plan, mrm_prompt_extraction,
                               mrm_prompt_final_answer,
                               mrm_prompt_perform_plan)
from utils import get_collection

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


model = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model="gpt-3.5-turbo",
            temperature=0,
            
        )
ef  =  embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name= "text-embedding-large",  #"text-embedding-large",
            api_base=api_base ,   #If you are using another service provider, please add the `api_base` parameter.
        )





design_plan_chain = (
            {
                "question": lambda x: x["question"]
            }
            | mrm_prompt_design_plan
            | model
            | StrOutputParser()
        )
perform_plan_chain = (
            {
                "knowledge": lambda x: x["knowledge"],
                "plan":  lambda x: x["plan"],
            }
            | mrm_prompt_perform_plan
            | model
            | StrOutputParser()
        )
extraction_knowledge_chain = (
            {
                "documents": lambda x: x["documents"],
                "question":  lambda x: x["question"],
            }
            | mrm_prompt_extraction
            | model
            | StrOutputParser()
        )
final_answer_chain = (
            {
                "all_knowledge": lambda x: x["all_knowledge"],
                "original_question":  lambda x: x["original_question"],
                "plan":  lambda x: x["plan"],
            }
            | mrm_prompt_final_answer
            | model
            | StrOutputParser()
        )

class SRM(object):
    def __init__(self):
        self.design_chain = design_plan_chain
        self.perform_chain = perform_plan_chain
        self.extraction_chain = extraction_knowledge_chain
        self.final_chain = final_answer_chain
        self.collection = get_collection(chroma_path=chroma_path,ef=ef)
    def search_documents_and_extract_knowledge(self,query):
        query = json.loads(query)
        restriction = query["subject"]
        question = query["question"]
        result = self.collection.query(
            query_texts=[question],
            where_document={"$contains": restriction},
            n_results=5,
        )
        documents = result["documents"][0]
        if len(documents) == 0:
            result = self.collection.query(
            query_texts=[query],
            n_results=5,
        )
            documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        knowledge =self.extraction_chain.invoke({"question":question,"documents":documents})
        return knowledge,documents,metadatas
    def parser_subject_question(self,plan):
        subject_question = plan.split("subject_question:")[1].strip()
        remove_sq_from_plan = plan.split("subject_question:")[0].strip()
        return subject_question,remove_sq_from_plan
    def parser_plan_step(self,plan):
        start_index = plan.find("Plan:")
        end_index = plan.find("First") 
        if start_index != -1 and end_index != -1:
            extracted_part = plan[start_index + len("Plan:"):end_index]
            num = extracted_part.strip()
            num = int(num)
            return num
    def mak_question_answer_file(self,file_path):
        folder_path = "./standard_question_answer"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"{folder_path} have been established")
        question_answer_file = os.path.basename(file_path).split(".")[0]
        question_answer_file= question_answer_file +".jsonl"
        question_answer_file = os.path.join(folder_path,question_answer_file)
        if not os.path.isfile(question_answer_file):
            f_save = open(question_answer_file,"w",encoding="utf-8")
            with open(file_path,"r",encoding="utf-8") as f_read:
                for i in f_read.readlines():
                    i_dict = json.loads(i)
                    del i_dict["paragraphs"]
                    del i_dict["question_decomposition"]
                    json.dump(i_dict,f_save,ensure_ascii=False)
                    f_save.write("\n")
            f_save.close()
        return question_answer_file
    def run(self,file_path):
        basename = os.path.basename(file_path)
        f_answer = open(os.path.join("./result",basename),"w",encoding="utf-8")
        question = []
        question_file = self.mak_question_answer_file(file_path)
        
        with open(question_file,"r",encoding='utf-8') as f:
            for i in f.readlines():
                i_dict = json.loads(i)
                question.append(i_dict)
    
        for i in question:
            try:
                plan_list= []
                knowledge_list = []
                metadata_list=[]
                documents_list = []
                plan = self.design_chain.invoke({"question":i["question"]})
                print(plan)
                plan_list.append(plan)
                num_step = self.parser_plan_step(plan)
                for _ in range(num_step-1):
                    subject_question, remove_sq_from_plan = self.parser_subject_question(plan)
                    knowledge,documents,metadatas = self.search_documents_and_extract_knowledge(subject_question)
                    knowledge_list.append(knowledge)
                    documents_list.append(documents)
                    metadata_list.append(metadatas)
                    plan = self.perform_chain.invoke({"knowledge":knowledge,"plan":remove_sq_from_plan})
                    plan_list.append(plan)
                plan_list.append(plan)
                subject_question,remove_sq_from_plan = self.parser_subject_question(plan)
                knowledge,documents,metadatas = self.search_documents_and_extract_knowledge(subject_question)
                knowledge_list.append(knowledge)
                documents_list.append(documents)
                metadata_list.append(metadatas)
                model_answer = self.final_chain.invoke({"all_knowledge":knowledge_list,"original_question":i["question"],"plan":remove_sq_from_plan})
                i["model_answer"]=model_answer
                i["knowledge_list"] = knowledge_list
                i["plan_list"] = plan_list
                i["metada"] = metadata_list
                i["documents"] = documents_list
                json.dump(i,f_answer,ensure_ascii=False)
                f_answer.write("\n")
            except Exception as e:
                print(e)
        f_answer.close()
            
if __name__ == "__main__":
    musique = SRM()
    musique.run(file_path)
