from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template = 'Generate a joke on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate the explanation of the following joke:\n {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = chain.invoke({'topic':'AI'})
print(result)