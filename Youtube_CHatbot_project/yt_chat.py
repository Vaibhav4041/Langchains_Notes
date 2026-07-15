from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter, Language
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os


# Step1a : Indexing (Document Ingestion)


video_id = '0jspaMLxBig'

try:
    # 1. Initialize the API instance
    api = YouTubeTranscriptApi()

    # 2. Get the transcript object (using the 2026 .fetch shortcut)
    # This returns a FetchedTranscript object
    fetched_transcript = api.fetch(video_id, languages=['en'])

    # 3. CONVERT to raw data (list of dicts) to fix your error
    data = fetched_transcript.to_raw_data()

    # 4. Now you can flatten it into text successfully
    transcript_text = " ".join(chunk['text'] for chunk in data)

    # print("Success! Transcript length:", len(transcript_text))

except Exception as e:
    print(f"An error occurred: {e}")

# Step1b : Indexing (Text Spliting)

spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = spliter.create_documents([transcript_text])

# print(len(chunks))

# print(chunks[50])

# Step1c and d : Indexing(Embeding Generation and storing in vector)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have a GPU
)

vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)

# print(vector_store.get_by_ids(['1fc6585e-0e4f-4225-a4eb-752ff7226304']))

# Step 2 : Retrieval

# retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs ={'k':3})

retriever = vector_store.as_retriever(
    search_type='similarity', 
    search_kwargs={'k': 3},
    tags=['FAISS', 'HuggingFaceEmbeddings'] # Overwrite the default tags here
)


# VectorStoreRetriever(tags=['FAISS', 'OpenAIEmbeddings'], vectorstore=<langchain_community.vectorstores.faiss.FAISS object at 0x7fdba029d2d0>, search_kwargs={'k': 4})


# print(retriever.invoke('Future of ai'))


# Step 3 Augmentation

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="chat-completion",
    temperature=0.01, # Set slightly above 0 for some providers
    max_new_tokens=500,
    # ADD THIS LINE: Force it to use Hugging Face's own infrastructure
    provider="hf-inference", 
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "Future of AI jobs"
retrieved_docs    = retriever.invoke(question)

print(retrieved_docs) 

print('#' * 80)

context_text = '\n\n'.join(doc.page_content for doc in retrieved_docs)

print('Context text ')
print('?'*80)
print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})

print('!'*80)
print(final_prompt)


# Step 4 generation

answer = llm.invoke(final_prompt)
print('Answer '*10)
print(answer.content)