import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from uuid import uuid4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from chromadb.errors import InvalidCollectionException


class ModelConfig:
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url

    def embedding_model(self):
        return DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=self.api_key)

    def chroma_ef(self):
        return embedding_functions.OpenAIEmbeddingFunction(api_key=self.api_key, api_base=self.base_url, model_name="text-embedding-v3")

    def llm_model(self):
        return init_chat_model(model="deepseek-v3", model_provider="openai", api_key=self.api_key, base_url=self.base_url)


class KnowledgeBaseConfig:
    def __init__(self, chroma_ef, embed_model):
        self.chroma_ef = chroma_ef
        self.embed_model = embed_model

    def create_kb(self, file_path: str, doc_size: int, doc_overlap: int):
        loader = PyMuPDFLoader(file_path, mode="single")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=doc_size, chunk_overlap=doc_overlap)
        all_splits = text_splitter.split_documents(docs)
        docs_with_content = [split if split.page_content else None for split in all_splits]
        uuids = [str(uuid4()) for _ in range(len(docs_with_content))]
        return uuids, docs_with_content

    def get_or_create_kb(self, fp="kb_files/优势谈判：适用于任何场景的经典谈判.pdf", doc_size=1000, doc_overlap=100):
        persistent_client = chromadb.PersistentClient(path="audio_copilot_db")
        try:
            collection = persistent_client.get_collection("negotiation_skill", embedding_function=self.chroma_ef)
        except InvalidCollectionException:
            collection = persistent_client.create_collection("negotiation_skill", embedding_function=self.chroma_ef)
            uuids, docs = self.create_kb(fp, doc_size=doc_size, doc_overlap=doc_overlap)
            batch_size = 10
            for i in range(0, len(docs), batch_size):
                batch_docs = docs[i:i + batch_size]
                batch_uuids = uuids[i:i + batch_size]
                collection.add(documents=[doc.page_content for doc in batch_docs],
                               metadatas=[doc.metadata for doc in batch_docs],
                               ids=batch_uuids)

        return Chroma(
            client=persistent_client,
            collection_name="negotiation_skill",
            embedding_function=self.embed_model,
        )




if __name__ == "__main__":
    load_dotenv()

    all_model_config = ModelConfig(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_BASE_URL"))
    embeddings = all_model_config.embedding_model()
    chroma_ef = all_model_config.chroma_ef()
    llms = all_model_config.llm_model()

    kb = KnowledgeBaseConfig(chroma_ef, embeddings)
    vec_store = kb.get_or_create_kb("kb_files/优势谈判：适用于任何场景的经典谈判.pdf", doc_size=1000, doc_overlap=100)