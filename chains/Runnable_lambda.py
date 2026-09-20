from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def word_count(word):
    return len(word.split())

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt = PromptTemplate(
    template = 'Generate a joke on the {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model,parser)

parllel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'Word_count':RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parllel_chain)
result = final_chain.invoke({'topic':'Cricket'})

final_result = """ {}\n word count is {}""".format(result['joke'],result['Word_count'])
print(final_result)