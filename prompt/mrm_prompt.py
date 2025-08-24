from langchain_core.prompts.prompt import PromptTemplate
mrm_design_plan_template="""
##Instruction
Make a plan for the given question. Pay attention to the nouns(eg. name,city,places) that appear in the question becauess they are the represent the subjects of the knowledge you desire to obtain. Don't use pronuns as subjects.

## Format
You can only respond with a single complete "Question understanding, Plan, suject_question" format.

## Subject Number
The subject number may be more than one. In the most situations, the number is One.

##Example
Q:How many private schools are in one of the oldest settlements in North America?
A: Question understanding: the question is about the number
Plan: 2
First: I need to know the oldest settlement in North America by North America
Second: I need to the number of private schools in the oldest settlement
subject_question:{{"subject":"North America","question":"which city is the oldest settlement in North America?"}}




Q: What is the maximum goal scored by a player in the event that was the first HDTV broadcast in Europe?
A: Question understanding: the question is about the number
Plan: 2
First: I need to know the event that was the first HDTV broadcast in Europe by HDTV broadcast
Second: I need to know the maximum goal scored by a player in the ev
subject_question:{{"subject":"HDTV broadcast","question":"what event was the first HDTV broadcast in Europe?"}}


Q: How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran-Iraq war?
A:Question understanding: This question is about the number. 
Plan:4
First:I need to the country which Pacific National University located in by  Pacific National University.
Second:I need to know the continent which the country located in by the country
Third: I need to know the the organization that mediated the truce ending the Iran-Iraq war by Iran-Iraq war.
Fourth: I need to know the number of country that the organization recognized in the continent by the organization.
subject_question:{{"subject":"Pacific National University","question":"which country is Pacific National University located in?"}}



Q: Who was born first, Dario Hübner or Luca Toni?
A: Question understanding: the question is about the person
Plan: 2
First: I need to know the date of birth of Dario Hübner and Luca Toni by Dario Hübner and Luca Toni.
Second: I need to know the who born first by the date of birth of Dario Hübner and Luca Toni
subject_question:{{"subject":"Dario Hübner, Luca Toni ","question":"when was Dario Hübner born?"}}



Q: In April 2015, how many death row inmates awaited execution in the state whose capital is the city where season 11 of Property Brothers was filmed?
A: Question understanding: the question is about the number
Plan: 3
First: I need to know the city where season 11 of Property Brothers was filmed by Property Brothers
Second:  I need to know which state the city is the capital of by the city
Third: I need to know the number of death row inmates awaiting execution in the state by the state.
subject_question:{{"subject":"Property Brothers","question":"where was season 11 of Property Brothers was filmed?"}}


Q: What is the acronym for the statewide criminal investigation agency whose capitol is in the city where Tree International Publishing was located?
A: Question understanding: the question is about the number
Plan: 3
First: I need to know the city that Tree International Publishing was located by Tree International Publishing
Second: I need to know which state the city is the capital of by the city
Third: I need to know to know  the acronym for the statewide criminal investigation agency of the state by the state.
subject_question:{{"subject":"Tree International Publishing","question":"what city was Tree International Publishing located in? "}}


Q: How long did the Great Irish Famine cause a population decline in the country of the singer of the album Live at the Point?
A: Question understanding: the question is about time
Plan: 3
First: I need to know the singer of the album Live at the Point by Live at the Point
Second: I need to know the country of the singer by the singer
Third: I need to know how long the Great Irish Famine cause a population decline in the country 
subject_question:{{"subject":"Live at the Point","question":"what singer of the album Live at the Point?"}}


Q:{question}

"""
mrm_perform_plan_template="""

##Instruction
Perfor a step of the OLD plan AND make a new plan. View the second step of the old plan as the the first step in the new plan.

You can only respond with a single complete

"Question understand, Have know, Plan, suject_question" format to make a new plan.



## Complete format:

Question understanding:

Have know:

Plan:num

(your new plan's step, use First, Second,etc)

suject_question:{{"subject":"subject","question":"question"}}

##Example
Old plan:
Knowledge: The oldest settlement in North America is St. John's
Question understanding: the question is about the number
Plan: 2
First: I need to know the oldest settlement in North America by North America
Second: I need to the number of private schools in the oldest settlement

New plan: Question understanding: the question is about the number
Have known: The oldest settlement in North America is St. John's
Plan: 1
First: I need to the number of private schools in St. John's by St. John's
subject_question:{{"subject":"St. John's","question":"how many private schools are in St. John's"?}}


Old plan: 
Knowledge: The band that performs Peace Sells is Megadeth
Question understanding: the question is about the number
Plan: 2
First: I need to know the band that performs Peace Sells by Peace Sells
Second: I need to know how many records the band has sold worldwide by the band

New plan: Question understanding: the question is about the number
Have known: The band that performs Peace Sells is Megadeth
Plan: 1
First: I need to know how many records Megadeth has sold worldwide by Megadeth
subject_question:{{"subject":"Megadeth","question":"how many records Megadeth has sold worldwide?"}}


Old plan:
Knowledge: The event that was the first HDTV broadcast in Europe is FIFA World Cup
Question understanding: the question is about the number
Plan: 2
First: I need to know the event that was the first HDTV broadcast in Europe by HDTV broadcast
Second: I need to know the maximum goal scored by a player in the ev

New plan: Question understanding: the question is about the number
Have known: The event that was the first HDTV broadcast in Europe is FIFA World Cup
Plan: 1
First: I need to know the maximum goal scored by a player in the FIFA World Cup
subject_question:{{"subject":"FIFA World Cup","question":"what is the maximum goal scored by a player in the FIFA World Cup?"}}


Old plan:
Knowledge: The planet that Budh Planitia located in is Mercury
Question understanding: the question is about time
Plan: 2
First: I need to know the planet that Budh Planitia located in by Budh Planitia
Second: I need to know the time it takes for the planet to orbit the sun by the planet.

New plan: Question understanding: the question is about time
PLan: 1
First: I need to know the time it takes for Mercury to orbit the sun by Mercury
subject_question:{{"subject":"Mercury","question":"how long does it take for Mercury to orbit the sun?"}}


Old plan:
Knowledge: The city is Nashville.
Question understanding: the question is about the number
Plan: 3
First: I need to know the city where season 11 of Property Brothers was filmed by Property Brothers
Second: I need to know the state whose capital is the city by the city
Third: I need to know the number of death row inmates awaiting execution in the state by the state.

New plan: Question understanding: the question is about the number
Have known: The city is Nashville.
Plan: 2
First: I need to know which state Nashville is the capital of by Nashville
Second: I need to know the number of death row inmates awaiting execution in the state by the state.
subject_question:{{"subject":"Nashville","question":""Which state's capital is Nashville?"}}


Old plan:
Knowledge: The city is Nashville.
Question understanding: the question is about acronym
Plan: 3
First: I need to know the city that Tree International Publishing was located by Tree International Publishing
Second: I need to know which state the city is the capital of by the city
Third: I need to know to know  the acronym for the statewide criminal investigation agency of the state by the state.

New plan: Question understanding: the question is about acronym
Have known: The city is Nashville.
Plan: 2
First: I need to know which state Nashville is the capital of by Nashville
Second: I need to know to know  the acronym for the statewide criminal investigation agency of the state by the state.
subject_question:{{"subject":"Nashville","question":"Which state's capital is Nashville?"}}


Old plan:
Knowledge: The singer is Christy Moore.
Question understanding: the question is about time
Plan: 3
First: I need to know the singer of the album Live at the Point by Live at the Point
Second: I need to know the country of the singer by the singer
Third: I need to know how long the Great Irish Famine cause a population decline in the country 

New plan: Question understanding: the question is about time
Have known: The singer is Christy Moore.
Plan: 2
First: I need to know the country of Christy Moore by Christy Moore
Second: I need to know how long the Great Irish Famine cause a population decline in the country by the country
subject_question:{{"subject":"Christy Moore","question":"which country is Christy Moore from?"}}


Old plan:
Knowledge: The country of Christy Moore is Ireland.
Question understanding: the question is about time
Have known: The singer is Christy Moore.
Plan: 2
First: I need to know the country of Christy Moore by Christy Moore
Second: I need to know how long the Great Irish Famine cause a population decline in the country by the country

New plan: Question understanding: the question is about time
Have known: The singer is Christy Moore. The country of Christy Moore is Ireland.
Plan: 1
First: I need to know how long the Great Irish Famine cause a population decline in Ireland  by Ireland
subject_question:{{"subject":" Ireland","question":"how long did the Great Irish Famine cause a population decline in Ireland?"}}


Old plan:
Knowledge: {knowledge}
{plan}

New plan:  
"""

mrm_extraction_template="""
Your task is to answer the question.Answer length should must more than 30 words.
Q:{question}
A:
"""

mrm_final_answer_template="""
{all_knowledge}
{original_question}
{plan}

"""
mrm_final_answer = PromptTemplate(
    template=mrm_final_answer_template,
    input_variables=["all_knowledge","original_question","plan"]
)
mrm_design_plan = PromptTemplate(
    template=mrm_design_plan_template,
    input_variables=["question"]
)
mrm_perform_plan = PromptTemplate(
    template=mrm_perform_plan_template,
    input_variables=["knowledge","plan"]
)
mrm_extraction = PromptTemplate(
    template=mrm_extraction_template,
    input_variables=["question"]
)