from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='provide a single appropriate response to this positive feedback.\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='provide a single appropriate response to this negative feedback.\n{feedback}',
    input_variables=['feedback']
)

parser1 = StrOutputParser()

classifier_chain = prompt1 | model | parser2

branch_chain =RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2 | model | parser1),
    (lambda x:x.sentiment =='negative', prompt3 | model | parser1),
    RunnableLambda(lambda x:'Couldnot find the sentiment from the feedback')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':'this is a beautiful smartphone'})
print(result)

