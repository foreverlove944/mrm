from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json
from tqdm import tqdm
from concurrent import futures
from threading import Thread,Lock
import time
import os
# os.environ["OPENAI_API_KEY"] = "sk-DCkz00L2RsVxPq4QE9Cc8111F6B041A999F454A1267eF78a"
# os.environ["OPENAI_API_KEY"] = "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"
evaluate_template = """\
Are the following two answers to the given question equivalent? Do not consider whether the answers are right or wrong. Do not consider whether the answers are in the same format.The only thing you need to consider is whether the answers are equivalent. Directly state ”Yes” or ”No”. 
Question: Which title was conferred to Anna Muzychuk in 2007? 
Answer 1: Anna Muzychuk was conferred the title of International Master (IM) in 2007. She earned the title by scoring three norms in rapid chess tournaments. 
Answer 2: International Master 
Answer 1 (short): International Master 
Answer 2 (short): International Master 
Are the two answers equivalent? 
Yes 
Question: What state is Seattle located in? 
Answer 1: Seattle is in Washington State. 
Answer 2: The answer is George Washington
Answer 1(short): George Washington
Answer 2(short): George Washington
Are the two answers equivalent? 
No 


Question: {question}
Answer 1: {model_output} 
Answer 2: {target_label}
Are the two answers equivalent?

"""

standard_answer_path="D:\\allcode\jiangnan-llm\\base_line\ircot-main\\a_zyq_experiment\musique\standard_id_answer.jsonl"
model_answer_path="glm4_result_no_intruction.jsonl"
evaluate_result_path="glm4_result_no_intruction_evaluate.jsonl"
evaluate_prompt = PromptTemplate(
    template=evaluate_template,
    input_variables=["question","model_output","target_label"]
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1,
    openai_api_base=r"https://uiuiapi.com/v1",
    openai_api_key=""
    
    
)
#openai_api_base=r"https://one.aiskt.com/v1",
evaluate_llm_chain = (
            {
              "question": lambda x: x["question"],
              "model_output":lambda x : x["model_output"],
              "target_label":lambda x : x["target_label"]
            }
            | evaluate_prompt
            | model
            | StrOutputParser()
        )
def us_model(data,file,lock):
    standard_answer = data[0]
    model_answer=data[1]
    try:
        if (standard_answer["id"] == model_answer["id"]):
            question = standard_answer["question_text"]
            model_output = model_answer["answer"]
            target_label =standard_answer["answer"][0]
            result =evaluate_llm_chain.invoke({"question":question,"model_output":model_output,"target_label":target_label})
            result_dict={"evaluate":result,"standard_answer":target_label}
            result_dict.update(model_answer)
            print(result)
            with lock:
                json.dump(result_dict,file,ensure_ascii=False)
                file.write("\n")    
    except Exception as e:
        print(e)


def get_answer_data(data_path):
    data=[]
    with open(data_path,"r",encoding="utf-8") as f:
        try:
            num = 0
            for i in f:
                num =num +1
                if i != '':
                    i = i.strip()
                    data.append(json.loads(i))
            return data
        except:
            print(num)
            print(i)
def evaluate():
    global evaluate_llm_chain
    standard_answer= get_answer_data(standard_answer_path)
    model_answer = get_answer_data(model_answer_path)
    futures_list = []
    executor = futures.ThreadPoolExecutor(max_workers=2)
    lock = Lock()
    file = open(evaluate_result_path,"a+",encoding="utf-8")
    file.seek(0)
    have_evaluated =set()
    for i in file:
        if i != "\n":
            i = json.loads(i)
            have_evaluated.add(i["id"])
    for i in tqdm(model_answer):
        id=i["id"]
        try:
            if id not in have_evaluated:
                id_standard_answer=next(filter(lambda x:x["id"]==id,standard_answer))
                futures_list.append(executor.submit(us_model,(id_standard_answer,i),file,lock))
        except Exception as e:
            print(e)
    completed, incomplete = futures.wait(futures_list, return_when=futures.ALL_COMPLETED)
    file.close()
evaluate()