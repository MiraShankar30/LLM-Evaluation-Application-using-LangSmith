from bs4 import BeautifulSoup as Soup
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load Documents
url = "https://drhyman.com/blogs/content/supplements-101-essential-vitamins-and-minerals"
loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=lambda x: Soup(x,"html.parser").text)
docs = loader.load()

# Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 1})

# Function to Get Answer
def get_answer(question: str) -> str:
    similar_docs = retriever.invoke(question)
    similar_text = "\n\n".join([str(doc) for doc in similar_docs])
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    # System prompt
    system_msg = (
        f"""
            You are a helpful AI code assistant with expertise in LangChain Expression Language (LCEL).
            Use the following context to answer the user's question. If the answer cannot
            be found in the context, please respond with "I cannot find the answer in the
            provided information."

            Context:{similar_text}
            Question: {question}
        """
    )

    messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": question},
    ]
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=google_api_key)
    response = llm.invoke(messages)

    return {
            "answer": response.content,
            "contexts": [str(doc) for doc in similar_docs],
            }
