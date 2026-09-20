from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents = ["Japan has developed a highly advanced railway network, with high-speed Shinkansen trains connecting major cities across the country.",
             
"Brazil is one of the world's largest agricultural producers, exporting large quantities of soybeans, coffee, sugar, and beef.",

"Egypt's economy and culture have been strongly influenced by the Nile River, which supports agriculture and provides water to much of the population.",

"Switzerland is well known for its banking sector, Alpine landscapes, and tradition of political neutrality."]

embedding = HuggingFaceEndpointEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query("Which country has a major high-speed railway system?")

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]
# print(sorted(list(enumerate(scores)), key = lambda x:x[1]))

# using numpy to sort

score = np.argmax(scores)

print(documents[index])
print("score is :",score)

