import argparse
import json
import os
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from prompt.cot import cot_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./standard_question_answer/musique_examples.jsonl')
parser.add_argument('--api_key', type=str,help='openai api key')
parser.add_argument('--api_base', default=None)
args = parser.parse_args()
api_key = args.api_key
file_path = args.input_file
api_base = args.api_base


model = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model="gpt-3.5-turbo",
            temperature=0,
            
        )
chain = (
            {
                "question": lambda x: x["question"]
            }
            | cot_prompt
            | model
            | StrOutputParser()
        )



basename = os.path.basename(file_path)
basename = "cot_" + basename
f_answer = open(os.path.join("./result",basename),"w",encoding="utf-8")
question = []
with open(file_path,"r",encoding='utf-8') as f:
    for i in f.readlines():
        i_dict = json.loads(i)
        question.append(i_dict)
for i in tqdm(question):
    answer = chain.invoke({"question":i["question"]})
    i["model_answer"] = answer
    json.dump(i,f_answer,ensure_ascii=False)
    
    f_answer.write("\n")