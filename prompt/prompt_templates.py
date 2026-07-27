from langchain_core.prompts import ChatPromptTemplate

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader assessing relevance of a retrieved document to a user question. "
               "If the document contains keywords or meaning related to the question, grade it as 'yes'. "
               "Otherwise, grade it as 'no'. Answer ONLY with 'yes' or 'no'."),
    ("human", "Question: {question}\n\nDocument: {document}")
])

REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI optimizing a search query. The original query failed to find good results. "
               "Rewrite the following question to be a better search keyword string. Do not add conversational text."),
    ("human", "{question}")
])

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following context to answer the question. "
               "If you don't know the answer based on the context, say that you don't know.\n\nContext: {context}"),
    ("human", "{question}")
])
