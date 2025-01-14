PROMPT_PYTHON = """
Please answer the question according to the given table regarding [TITLE] 
The table context is: [TABLE]
The question is: [QUESTION]

Notes: 
- Give me the answer in format "Final Answer: AnswerName1, AnswerName2..." form (should be a number or entity names, as short as possible, without any explanation)
- Give me the python script (prefer using list operations instead of dataframe)
- Use print function to output the final answer
- For percentage calculations, keep two decimal places
- Give the process of deducing the final answer with intermediate results
"""

PROMPT_TEXT = """
Please answer the question according to the given table regarding [TITLE] 
The table context is: [TABLE]
The question is: [QUESTION]

Notes:
- Give me the answer in format "Final Answer: AnswerName1, AnswerName2..." form (should be a number or entity names, as short as possible, without any explanation)
- Give the process of deducing the final answer with intermediate results
"""

PROMPT_CLS = """
Below is a table header regarding "[TITLE]":
[HEADER]
You're tasked with answering the following question:
[QUESTION]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: [INST1]. 
Answer [A] is: [ANSWER1].
Reasoning of Answer [B] is: [INST2]. 
Answer [B] is: [ANSWER2].
Your task is to determine which is the correct answer. The final answer is [A] if Answer A is correct, and [B] if Answer B is correct.
If Answer [A] and Answer [B] are the same, the final answer could be either [A] or [B].
Therefore, the final answer is: [ALPHA]
"""

PROMPT_VERIF = """
Below is a table header regarding "[TITLE]":
[HEADER]
You're tasked with answering the following question:
[QUESTION]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: [INST1]. 
Answer [A] is: [ANSWER1].
Reasoning of Answer [B] is: [INST2]. 
Answer [B] is: [ANSWER2].
Your task is to determine whether this question can be correctly answered by these two models. True means yes and False means no.
Therefore, the final answer is: [ALPHA]
"""

PROMPT_CLS_TEST = """
Below is a table header regarding "[TITLE]":
[HEADER]
You're tasked with answering the following question:
[QUESTION]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: [INST1]. 
Answer [A] is: [ANSWER1].
Reasoning of Answer [B] is: [INST2]. 
Answer [B] is: [ANSWER2].
Your task is to determine which is the correct answer. The final answer is [A] if Answer A is correct, and [B] if Answer B is correct.
If Answer [A] and Answer [B] are the same, the final answer could be either [A] or [B].
Therefore, the final answer is:
"""

PROMPT_VERIF_TEST = """
Below is a table header regarding "[TITLE]":
[HEADER]
You're tasked with answering the following question:
[QUESTION]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: [INST1]. 
Answer [A] is: [ANSWER1].
Reasoning of Answer [B] is: [INST2]. 
Answer [B] is: [ANSWER2].
Your task is to determine whether this question can be correctly answered by these two models. True means yes and False means no.
Therefore, the final answer is:
"""