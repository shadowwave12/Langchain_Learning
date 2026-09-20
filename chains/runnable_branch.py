from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableBranch,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt1 = PromptTemplate(
    template = 'Generate a report on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following topic in less than 300 words \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# report_gen_chain = RunnableSequence(prompt1,model,parser)
report_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableBranch(

    (lambda x:len(x.split())>300, RunnableSequence(RunnableLambda(lambda x:{'text':x}),prompt2,model,parser)),
    RunnablePassthrough()
   
)

final_chain = RunnableSequence(report_gen_chain,parallel_chain, RunnableParallel({
        'word_count':RunnableLambda(lambda x:len(x.split())),
        'report': RunnablePassthrough()
    }))
result = final_chain.invoke({'topic':'cricket'})
print(result)