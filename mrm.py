import argparse
import json
import os
import copy
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from prompt.mrm_prompt import (mrm_final_answer,mrm_design_plan,mrm_perform_plan,mrm_extraction)


from utils import get_collection

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./standard_question_answer/musique_examples.jsonl')
parser.add_argument('--api_key', type=str,help='openai api key')
parser.add_argument('--chroma_path', default="./chroma_db", help='chroma db path')
parser.add_argument('--api_base', default=None)
parser.add_argument('--similarity_distance_threshold', default=0.8)

args = parser.parse_args()

api_key = args.api_key
file_path = args.input_file
chroma_path = args.chroma_path
api_base = args.api_base #"https://sg.uiuiapi.com/v1" #args.api_base
similarity_distance_threshold = float(args.similarity_distance_threshold)


model = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model="gpt-3.5-turbo",
            temperature=0,
            
        )
ef  =  embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name= "text-embedding-3-large",  
            api_base=api_base ,   #If you are using another service provider, please add the `api_base` parameter.  you can use https://sg.uiuiapi.com/v1

        )





design_plan_chain = (
            {
                "question": lambda x: x["question"]
            }
            | mrm_design_plan
            | model
            | StrOutputParser()
        )
perform_plan_chain = (
            {
                "knowledge": lambda x: x["knowledge"],
                "plan":  lambda x: x["plan"],
            }
            | mrm_perform_plan
            | model
            | StrOutputParser()
        )
extraction_knowledge_chain = (
            {
                "documents": lambda x: x["documents"],
                "question":  lambda x: x["question"],
            }
            | mrm_extraction
            | model
            | StrOutputParser()
        )
final_answer_chain = (
            {
                "all_knowledge": lambda x: x["all_knowledge"],
                "original_question":  lambda x: x["original_question"],
                "plan":  lambda x: x["plan"],
            }
            | mrm_final_answer
            | model
            | StrOutputParser()
        )

class MRM(object):
    def __init__(self):
        self.design_chain = design_plan_chain
        self.perform_chain = perform_plan_chain
        self.extraction_chain = extraction_knowledge_chain
        self.final_chain = final_answer_chain
        self.collection = get_collection(chroma_path=chroma_path,ef=ef)
        self.similarity_distance_threshold = similarity_distance_threshold
    

    def search_documents_and_extract_knowledge(self,query,similarity_distanbce_threshold=0.8):  # similarity_distanbce =1-similarity

        def split_string(string):
            if "," in string:
                subject_list = string.split(",")
                temp_list = []
                for i in subject_list:
                    if "'s" in string:
                        _subject_list = i.split("'s")
                        temp_list= temp_list+_subject_list
                    else:
                        temp_list.append(i)
                return temp_list
            if "'s" in string:
                subject_list = string.split("'s")
            else:
                subject_list = [string]
            return subject_list
        def replace_json(input_string):
            start_marker = "subject"
            end_marker = 'question'

            start_index = input_string.find(start_marker)
            end_index = input_string.find(end_marker)

            if start_index != -1 and end_index != -1:
                extracted_subject = input_string[start_index + len(start_marker) : end_index]
                extracted_subject = extracted_subject.strip(":,'\"")
                extracted_question = input_string[end_index + len(end_marker) :]
                extracted_question = extracted_question.strip(":,'}")
                extracted_question = extracted_question.replace('"', "")
                return extracted_subject, extracted_question

        query = query.strip(":,\n ")
        doc_list = []
        metadatas_list = []
        try:
            query = json.loads(query)
            restriction = query["subject"]
            question = query["question"]
        except Exception as e:
            restriction, question = replace_json(query)
        finally:
            try:
                subject_list = split_string(restriction)

                try:
                    for i in subject_list:
                        i = i.strip(":\n,' ")
                        get_retrieve = self.collection.query(query_texts=[question], n_results=5, where_document={"$contains": i})
                        if len(get_retrieve["documents"][0]) != 0:
                            for index,j in enumerate(get_retrieve["distances"][0]):
                                if j < similarity_distanbce_threshold:
                                    doc_list.append(get_retrieve["documents"][0][index])
                                    metadatas_list.append(get_retrieve["metadatas"][0][index])
                    if len(doc_list) == 0:
                        input = ",".join(subject_list) + "\n\n" + question
                        get_retrieve = self.collection.query(query_texts=[input], n_results=5)
                        if len(get_retrieve["documents"][0]) != 0:
                            for index,j in enumerate(get_retrieve["distances"][0]):

                                if j < similarity_distanbce_threshold:
                                    doc_list.append(get_retrieve["documents"][0][index])
                                    metadatas_list.append(get_retrieve["metadatas"][0][index])
                except Exception as e:
                    print("para error" + e)
                    input = ",".join(subject_list) + "\n\n" + question
                    get_retrieve = self.collection.query(query_texts=[input], n_results=5)
                    if len(get_retrieve["documents"][0]) != 0:
                        for index,j in enumerate(get_retrieve["distances"][0]):
                            if j < similarity_distanbce_threshold:
                                doc_list.append(get_retrieve["documents"][0][index])
                                metadatas_list.append(get_retrieve["metadatas"][0][index])

                finally:
                    processed_information = extraction_knowledge_chain.invoke(

                        {"documents": doc_list, "question": question}

                    )
                    # for i in range(len(doc_)):
                    #     title =doc_[i]
                    #     paragraph_text=metadatas_[i]["paragraph_text"]
                    #     got_doc=title +"\n" +paragraph_text
                    #     metadatas_[i].pop("paragraph_text")
                    #     doc.append(got_doc)
                    #     metadatas.append(metadatas_[i])

                    return processed_information, metadatas_list,doc_list
                # else:
                #     doc_,metadatas_ = query_from_vector(i)
                #     return doc_,metadatas_
            except Exception as e:
                    print("**************************parse_thought_retrieve**************\n")
                    print(e)
                    print("**************************parse_thought_retrieve**************\n")

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
    def run(self,file_path):
        basename = os.path.basename(file_path)
        basename = "mrm_" + basename
        f_answer = open(os.path.join("./result",basename),"w",encoding="utf-8")
        question = []
        with open(file_path,"r",encoding='utf-8') as f:
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
                plan_list.append(copy.deepcopy(plan))

                num_step = self.parser_plan_step(plan)
                for _ in range(num_step-1):
                    subject_question, remove_sq_from_plan = self.parser_subject_question(plan)
                    knowledge,metadatas,documents = self.search_documents_and_extract_knowledge(subject_question,self.similarity_distance_threshold)

                    knowledge_list.append(knowledge)
                    documents_list.append(documents)
                    metadata_list.append(metadatas)
                    plan = self.perform_chain.invoke({"knowledge":knowledge,"plan":remove_sq_from_plan})
                    plan_list.append(copy.deepcopy(plan))
                plan_list.append(copy.deepcopy(plan))

                subject_question,remove_sq_from_plan = self.parser_subject_question(plan)
                knowledge,metadatas,documents = self.search_documents_and_extract_knowledge(subject_question)
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
    musique = MRM()
    musique.run(file_path)
