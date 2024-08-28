## Default to using gpt-3.5-turbo
## Prepare the data
1. Down load the dataset from hugging face
Musique： https://huggingface.co/datasets/dgslibisey/MuSiQue

2. Randomly sample 2,000 questions

3. Vectorize documents from 2,000 questions by using text-embeding-large from OpenAI

    pthon vectorize_documents.py --input_file_path "./dataset/musique_examples.jsonl" --api_kye your_api_key --api_base your_api_base 
    Note: 
## Experiments
pthon srm.py --input_file_path standard_question_answer --api_kye your_api_key --api_base your_api_base 