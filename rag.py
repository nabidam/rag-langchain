from langchain import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

template = """                                                                                                                                                                                                                                   
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer c      oncise.                                                                                                                                                                                                                               
Question: {question}                                                                                                                                                                                                                  
Context: {context}                                                                                                                                                                                                                    
Answer:                                                                                                                                                                                                                               
"""

template = """                                                                                                                                                                                                                                   
You are a programming assistant for writing code tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, use your knowledge, if again you can't find any answer, just say that you don't know the answer.                                                                                                                                                                                                                               
Question: {question}                                                                                                                                                                                                                  
Context: {context}                                                                                                                                                                                                                    
Answer:                                                                                                                                                                                                                               
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

vectorstore = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

result = qa_chain.invoke({"query": "How can I build an application using langchain that reads a dataset in csv format, and analyze it? can you show me an example of code?"})

print(result)
