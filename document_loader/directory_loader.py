from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = DirectoryLoader(
    path = 'books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate a summary on the following text - \n {text}',
    input_variables=['text']
)

docs = loader.lazy_load()

for document in docs :
    print(document.metadata)

chain = prompt | model | parser

# chain.invoke({'text': docs[0].page_content})