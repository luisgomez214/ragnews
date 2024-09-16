This repository contains a Question and Answer (Q&A) system designed for answering questions. The system uses a Retrieval-Augmented Generation (RAG) approach, where it retrieves relevant articles from an SQL database and uses Groq’s LLM API to generate responses. The project demonstrates how traditional SQL databases can be used effectively for document retrieval without relying on complex vector databases like Pinecone.

```
(venv) luis@Luiss-MacBook-Pro-40 ragnews1 %  python3 ragnews.py --query "Who is the president"                              
According to the provided article summaries, there is no specific information about the current president of the United States. However, in one of the summaries, it is mentioned that Donald Trump made a statement during a debate, and in another summary, it is mentioned that he is the former President. 

It is likely that the article being referred to is not up to date, or it may be referring to Donald Trump, who was the 45th President of the United States from 2017 to 2021. If you are looking for more current information, I would suggest checking reputable news sources or official government websites for the most accurate and up-to-date information.
```

