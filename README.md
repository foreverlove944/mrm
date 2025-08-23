## Default to using gpt-3.5-turbo
## Prepare the data
1. Down load the dataset from hugging face
Musique： https://huggingface.co/datasets/dgslibisey/MuSiQue

2. Randomly sample 2,000 questions

3. Vectorize documents from 2,000 questions by using text-embeding-large from OpenAI

    pthon create_vec.py --api_kye your_api_key --api_base your_api_base 


    Note: you should provide your own api key and api base
## Experiments
pthon mrm.py --input_file_path question_file --api_kye your_api_key --api_base your_api_base 