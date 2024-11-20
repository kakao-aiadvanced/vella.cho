### Index
import os
import streamlit as st
from tavily import TavilyClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

tavily = TavilyClient(api_key='')
os.environ['OPENAI_API_KEY'] = ''
llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

### Router
# 사용자 질문을 벡터 저장소(vectorstore)나 웹 검색(web search)으로 라우팅하는 전문가입니다.
# 다음 기준을 따르세요:
# LLM 에이전트, 프롬프트 엔지니어링, 적대적 공격(adversarial attacks)에 대한 질문은 벡터 저장소(vectorstore)를 사용합니다.
# 이 주제와 관련된 키워드는 엄격히 따질 필요가 없습니다.
# 그 외의 모든 경우에는 웹 검색(web search)을 사용합니다.
# 결과는 web_search 또는 vectorstore 중 하나의 이진 선택으로 반환합니다.
# JSON 형식으로 단일 키 'datasource'만 포함하며, 설명이나 부연 설명은 작성하지 않습니다.

system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}"),
    ]
)

question_router = prompt | llm | JsonOutputParser()

question = "llm agent memory" #vectorstore
# question = "What is prompt?" #web_search
docs = retriever.get_relevant_documents(question)
# print(question_router.invoke({"question": question}))

### Retrieval Grader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 당신은 사용자 질문과 검색된 문서 간의 연관성을 평가하는 채점자입니다.
# 문서에 사용자 질문과 관련된 키워드가 포함되어 있다면 해당 문서를 관련 있음으로 평가합니다.
# 엄격한 테스트가 필요하지 않습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다.
# 결과는 이진 점수 yes 또는 no 로 제공하며,
# JSON 형식으로 단일 키 'score'만 포함하며, 부가적인 설명이나 서술은 작성하지 않습니다.
system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "What is prompt?"
# question = "who is vella?" #{'score': 'no'}
docs = retriever.invoke(question)
doc_txt = docs[0].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# 당신은 질문-응답 작업을 위한 어시스턴트입니다.
# 검색된 문맥을 사용하여 질문에 답하세요.
# 답을 모를 경우, "모르겠습니다."라고 간단히 말하세요.
# 최대 세 문장으로 답변을 작성하며, 답변은 간결하게 유지합니다.
system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
question = "What is prompt?"
# question = "2025년 11월 20일 삼성전자 주가는?" #I don't know.
docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

### Hallucination Grader
# 당신은 답변이 사실에 근거하거나 지원되는지를 평가하는 채점자입니다.
# 답변이 사실에 근거하거나 지원되는 경우, 이진 값 'yes' 또는 'no'로 점수를 부여하세요.
# 점수는 JSON 형식으로 단일 키 'score'로 제공하며, 추가 설명이나 서문 없이 작성하세요.
system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)

hallucination_grader = prompt | llm | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})

### Answer Grader
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Prompt
# 당신은 답변이 질문을 해결하는 데 유용한지를 평가하는 채점자입니다.
# 답변이 질문을 해결하는 데 유용한 경우, 이진 값 'yes' 또는 'no'로 점수를 부여하세요.
# 점수는 JSON 형식으로 단일 키 'score'로 제공하며, 추가 설명이나 서문 없이 작성하세요.
system = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})

### State


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    generation_count: int


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    print(documents)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation_count = state.get("generation_count", 0)

    # 2회 이상 동작하지 않도록 제한
    if generation_count >= 2:
        return {"documents": documents, "question": question, "generation": generation, "generation_count": generation_count}

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    generation_count += 1
    return {"documents": documents, "question": question, "generation": generation, "generation_count": generation_count}

def relevance_checker(state): # grade_documents -> relevance_checker
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    relevance_check = ""
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue

    if relevance_check == "No":
        print("failed: not relevant")

    # filtered_docs 이 0개면 web_search
    if len(filtered_docs) == 0:
        print("all documents are relevant")
        web_search = "Yes"
        relevance_check = "No"

    return {"documents": filtered_docs, "question": question, "web_search": web_search, "relevance_check": relevance_check}

    def web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        docs = tavily.search(query=question)["results"]
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    print(state)
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']

    # Create new documents with 'source' and 'title' instead of content
    results = []
    for d in docs:
        metadata = {
            "source": d["url"],  # Use URL as the source
            "title": d["title"]   # Use the title as the title
        }
        # Add the metadata as part of the document
        result = Document(page_content=d["content"], metadata=metadata)
        results.append(result)

    # Return the updated state with source and title stored
    if documents is not None:
        documents.extend(results)  # Append to existing documents
    else:
        documents = results  # If no documents, initialize with results

    return {"documents": documents, "question": question}


### Edges


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge

# grade_generation_v_documents_and_question -> hallucination_checker
def hallucination_checker(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    generation_count = state["generation_count"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade.lower() == "yes" and generation_count == 1:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "success"
    elif grade.lower() == "no" and generation_count > 1:
        print("failed: hallucination")
        return "failed"
    elif grade.lower() == "no" and generation_count == 1:
        print("---DECISION: DECISION: GENERATION IS HALLUCINATION, RE-TRY---")
        return "hallucination"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("relevance_checker", relevance_checker)  # grade documents
workflow.add_node("generate", generate)  # generatae

# Build graph
# workflow.set_conditional_entry_point(
#     route_question,
#     {
#         "websearch": "websearch",
#         "vectorstore": "retrieve",
#     },
# )

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "relevance_checker")
workflow.add_conditional_edges(
    "relevance_checker",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    hallucination_checker,
    {
        "failed": END,
        "success": END,
        "hallucination": "generate",
    },
)

# Compile
app = workflow.compile()

# Test

inputs = {"question": "What is prompt?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
print("(출처)")
print('\n'.join(set([f"URL: {d.metadata['source']}" for d in value['documents']])))
print('\n'.join(set([f"제목: {d.metadata['title']}" for d in value['documents']])))

# 여러 질문을 수행할거라면 document 등 state초기화 필요
document = None

from pprint import pprint

# Compile
app = workflow.compile()
inputs = {"question": "Where does Messi play right now?"}
# inputs = {"question": "한글로답해줘. 2024년 11월 20일자 신한알파리츠의 종가는 얼마야?"}
# inputs = {"question": "Who is vella?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])
print('\n')
print("(출처)")
print('\n'.join(set([f"URL: {d.metadata['source']}" for d in value['documents']])))
print('\n'.join(set([f"제목: {d.metadata['title']}" for d in value['documents']])))


# Streamlit 앱 UI
st.title("Research Assistant powered by OpenAI")

input_topic = st.text_input(
    ":female-scientist: Enter a topic",
    value="Superfast Llama 3 inference on Groq Cloud",
)

generate_report = st.button("Generate Report")

if generate_report:
    with st.spinner("Generating Report"):
        inputs = {"question": input_topic}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}:")
        final_report = value["generation"]
        st.markdown(final_report)

st.sidebar.markdown("---")
if st.sidebar.button("Restart"):
    st.session_state.clear()
    st.experimental_rerun()
