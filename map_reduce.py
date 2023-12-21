# Python built-in module
import os
import time
import json
import traceback

# Python installed module
import tiktoken
import langchain
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
from cod import COD
from langchain.llms import CTransformers
from sentence_splitter import SentencizerSplitter


class MapReduce():
    '''This class implements the Map Reduce summarization'''
    
    def __init__(self):
        self.llm =  llm = CTransformers(model="TheBloke/Llama-2-13B-chat-GGUF",
                    model_file='llama-2-13b-chat.Q4_K_S.gguf',
                    model_type='llama', 
                    config={'max_new_tokens': 1024, 'threads': 8, 'context_length': 2048}
        )
        
        kw_extraction_prompt_template = PromptTemplate(input_variables=["text_chunk"], template=prompts.KW_EXTRACT_SYSTEM_PROMPT)
        kw_extraction_chain = LLMChain(llm=self.llm, prompt=kw_extraction_prompt_template, output_key="key_words")
        
        summary_prompt_template = PromptTemplate(input_variables=["text_chunk", "key_words"], template=prompts.SEQUENCIAL_SUMMARY_PROMPT)
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt_template, output_key="summary")
        
        self.extractive_summary_chain = SequentialChain(chains=[kw_extraction_chain, summary_chain], input_variables=["text_chunk"],
                                                        output_variables=["key_words", "summary"], verbose=False
                                                       )
        
        reduce_prompt = PromptTemplate.from_template(prompts.REDUCE_PROMPT)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000,
        )
        self.text_splitter = SentencizerSplitter()
        self.chain_of_density_summarizer = COD()
        
    def __call__(self, text_content, redirect=None):
        try:
            start_time = time.time()
            print("[INFO] The Map Reduce summarization started...")
            summaries_list = list()
            do_cod = False
            print("[INFO] Text chunking started...")
            document_splits = self.text_splitter.create_documents(text_content)
            total_splits = len(document_splits)
            print("[INFO] Text chunking done!")
            print("[INFO] Map chain for extractive summarization started...")
            
            with get_openai_callback() as openai_cb:
                for idx, doc in enumerate(document_splits, start=1):
                    print("[INFO] Map chain extractive summarization for chunk: {}/{}".format(idx, total_splits))
                    extractive_summary_result = self.extractive_summary_chain({"text_chunk": doc.page_content})
                    summaries_list.append(extractive_summary_result["summary"])
                print("[INFO] Map chain for extractive summarization done!")
                print("[INFO] Reduce chain for summarization started...")
                final_summary = self.reduce_documents_chain.run([langchain.schema.document.Document(page_content=chunk) for chunk in summaries_list])
                print("[INFO] Reduce chain for summarization done!")
                # print("[INFO] Keywords extraction from final summary started...")
                # kw_extract_messages = [HumanMessage(content=prompts.KW_EXTRACT_SYSTEM_PROMPT.format(text_chunk=final_summary))]
                # kw_response = self.llm(kw_extract_messages)
                # kw_output = kw_response.content.split(", ")
                print()
                print(final_summary)
                # print(kw_output)
                print("[INFO] Keywords extraction from final summary done!")
                return {"summary": final_summary}            
        #     if (redirect == "cluster_summary") and (self.config_dict["cluster_summarization"]["final_dense"] == True):
        #         do_cod = True
        #     elif (redirect == "cluster_summary") and (self.config_dict["cluster_summarization"]["final_dense"] == False):
        #         do_cod = False
        #     elif (redirect is None) and (self.config_dict["cluster_summarization"]["final_dense"] == True):
        #         do_cod = True
        #     elif (redirect is None) and (self.config_dict["cluster_summarization"]["final_dense"] == False):
        #         do_cod = False
                
        #     if do_cod:
        #         cod_result_dict = self.chain_of_density_summarizer(final_summary)
        #         end_time = time.time()
        #         print("[INFO] The Map Reduce summarization started!")
        #         return {"summary": cod_result_dict["summary"],
        #                 "keywords": cod_result_dict["keywords"],
        #                 "metadata": {"total_tokens": openai_cb.total_tokens + cod_result_dict["metadata"]["total_tokens"],
        #                              "total_cost": round(openai_cb.total_cost, 3) + cod_result_dict["metadata"]["total_cost"],
        #                              "total_time": round((end_time-start_time), 2)}}
            
        #     end_time = time.time()
        #     print("[INFO] The Map Reduce summarization done!")
        #     return {"summary": final_summary,
        #             "keywords": kw_output }
        except Exception as error:
            print("[ERROR] Some error happend in Map Reduce. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return