from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
   def hello():
    print("Hello World")

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 52,
    chunk_overlap = 0
)

result = splitter.split_text(text=text)
print(len(result))
print(result)