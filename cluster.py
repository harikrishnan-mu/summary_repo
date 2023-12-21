# Python built-in module
import os
import time
import json
import traceback

# Python installed module
# import openai
import tiktoken
import langchain
import numpy as np
from sklearn.cluster import KMeans
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

# Python user defined module
import prompts
from map_reduce import MapReduce
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_splitter import SentencizerSplitter
from map_reduce import MapReduce

def cluster_based_summary(text_content):
    embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'})
    embedding_chunk_size=10
    num_clusters=4
    k=1
    map_reduce_summarizer = MapReduce()
    try:
        with get_openai_callback() as openai_cb:
            start_time = time.time()
            print("[INFO] Cluster based summarization started...")
            print("[INFO] Text chunking started...")
            document_splits = SentencizerSplitter().create_documents(text_content)
            total_splits = len(document_splits)
            print("[INFO] Text chunking done!")
            print("[INFO] Text embedding started...")
            vectors = embeddings.embed_documents(texts=[x.page_content for x in document_splits])
            print("[INFO] Text embedding done!")
            print("[INFO] K-Means clustering started, this might take some time...")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
            print("[INFO] K-Means clustering done!")
            print("[INFO] Finding the closest points to each cluster center")
            closest_indices = []
            for i in range(num_clusters):
                distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                closest_index = np.argsort(distances)[:k]
                closest_indices.extend(closest_index)
                selected_indices = sorted(closest_indices)
            sorted_documents = [document_splits[idx].page_content for idx in selected_indices]
            mr_result_dict = map_reduce_summarizer("\n".join(sorted_documents), redirect="cluster_summary")
            end_time = time.time()
            print("[INFO] Cluster based summarization done!")
            
        return {"summary": mr_result_dict["summary"],
                "keywords": mr_result_dict["keywords"]}
        
    except Exception as error:
        print("[ERROR] Some error happend in Map Reduce. Error:\n\n{}\n\n".format(error))
        traceback.print_exception(type(error), error, error.__traceback__)
        return