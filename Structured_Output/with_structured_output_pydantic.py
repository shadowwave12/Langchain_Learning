from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional,Literal
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm =llm)

#schema
class Review(BaseModel):
    summary: str = Field(description='write a brief summary of the review')
    sentiment: Literal["pos","neg"]= Field(description='Return sentiment of the review')
    name : Optional[str]= Field("write a name of the reviewer")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.
 There are too many pre-installed apps that I can't remove. 
 Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
# print(result['summary'])
# print(result['sentiment'])