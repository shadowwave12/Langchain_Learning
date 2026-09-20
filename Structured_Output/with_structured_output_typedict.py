from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    provider="auto"
)

model = ChatHuggingFace(llm =llm)

#schema
class Review(TypedDict):
    summary : Annotated[str,"write a brief summary of the review"]
    sentiment : str
    name : Annotated[Optional[str],"write a name of the reviewer"]


structured_model = model.with_structured_output(
    Review,
    method="function_calling"
)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.
 There are too many pre-installed apps that I can't remove. 
 Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
# print(result['summary'])
# print(result['sentiment'])