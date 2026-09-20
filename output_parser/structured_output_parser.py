from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = 'fact1', description = 'fact 1 about the topic'),
    ResponseSchema(name = 'fact2', description = 'fact 2 about the topic'),
    ResponseSchema(name = 'fact3', description = 'fact 3 about the topic')

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='give 3 facts on /n {topic}',
    input_variables= ['topic']
)

chain = template | model | parser

result = chain.invoke({'topic':'blackhole'})
print(result)