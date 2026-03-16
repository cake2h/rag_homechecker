import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-..." 

PERSIST_DIRECTORY = "./db_storage"

def load_and_process_docs(docs_path):
    """Загружает документы из папки и нарезает на чанки."""
    documents = []
    # Здесь логика загрузки разных форматов
    for file in os.listdir(docs_path):
        path = os.path.join(docs_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(splits):
    """Создает или обновляет векторную базу."""
    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function, 
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

def get_qa_chain(vectorstore):
    """Создает цепочку для проверки работ."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # Температура 0 для объективности
    
    # Промпт — самое важное место
    template = """
    Ты строгий, но справедливый преподаватель. Твоя задача — проверить домашнюю работу студента.
    
    Контекст (методические материалы и критерии):
    {context}
    
    Задание студента:
    {question}
    
    Инструкция:
    1. Сравни ответ студента с контекстом.
    2. Укажи на фактические ошибки.
    3. Оцени полноту ответа.
    4. Предложи конкретные улучшения.
    5. Если работа выполнена идеально, похвали студента.
    
    Ответ оформи в формате:
    - Оценка (0-10):
    - Обнаруженные ошибки:
    - Рекомендации:
    - Итоговый комментарий:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Берем топ-5 релевантных кусков
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain