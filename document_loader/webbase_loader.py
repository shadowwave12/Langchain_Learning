from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


url = 'https://www.amazon.com/Apple-MacBook-13-inch-Storage-Midnight/dp/B0DBX8SM11/ref=sr_1_1?dib=eyJ2IjoiMSJ9.XCemkdggK7U5ltpNSMVdgLs6j8UQihkj9zoysbbbq6sa233VEJGjuNNZijZzTXyp06WHRN46OHraAAYheDa0A4UkT704DLNFlIv4nfda1nXJO28MhIbvGcJoxHga2ve_H3yiNeh4PwB0TDE9CI_Zq-6WjVkglpTUS-LhBs5db1e0GTKxEpv_lyncff13M13f4DQg1TQyUbSJ5I2LwMmdPKwbKYCfWTpKSV-O_YhOWBk.sn7mqZDkobXsDd8Bp7ln3uacNtplJ1eiJOmp1uvUuuE&dib_tag=se&keywords=macbook%2Bair&qid=1789531440&sr=8-1&th=1'
loader = WebBaseLoader(url)

docs = loader.load()
print(len(docs))
# print(docs[0].page_content)

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question - \n {question} on the following text - \n {text}',
    input_variables=['question','text']
)

chain = prompt | model | parser
result = chain.invoke({'question':'What is the name of the product?', 'text' : docs[0].page_content})

print(result)