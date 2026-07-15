# from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

loader = PyPDFLoader('Machine Learning Questions.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_documents(docs)

print(result[1].page_content)