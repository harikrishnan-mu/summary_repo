# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.chains.summarize import load_summarize_chain
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import pipeline 
# import torch
# import base64
# from langchain.llms import CTransformers


# config = {'max_new_tokens': 1024, 'context_length': 2048, 'temperature': 0.01, 'threads': 8}
# llm = CTransformers(model="TheBloke/CodeLlama-13B-Instruct-GGUF",
#                     model_file='codellama-13b-instruct.Q4_K_M.gguf',
#                     model_type='llama', config=config)


# def file_preprocessing(file):
#     loader = PyPDFLoader(file)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     texts = text_splitter.split_documents(pages)
#     final_texts = ''
#     for text in texts:
#         final_texts = final_texts + text.page_content
#     return final_texts

# def llm_pipeline(filepath):
#     pipe_sum = pipeline('summarization',
#     model = llm,
#     # tokenizer = tokenizer,
#     max_length=500,
#     min_length= 100
#     )
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# result=llm_pipeline('data/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone.pdf')


from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import CTransformers
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

config = {'max_new_tokens': 1024, 'context_length': 2048, 'temperature': 0.01, 'threads': 8}
llm = CTransformers(model="TheBloke/Llama-2-13B-chat-GGUF",
                    model_file='llama-2-13b-chat.Q4_K_S.gguf',
                    model_type='llama', config=config)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)


# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=1000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)


loader = PyPDFLoader('data/Joanne-K.-Rowling-Harry-Potter-Book-1-Harry-Potter-and-the-Philosophers-Stone.pdf')
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=10
)
split_docs = text_splitter.split_documents(docs)
print("_________")
print(map_reduce_chain.run(split_docs))


