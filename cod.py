import os
import time
import json
import traceback

# Python installed module
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import CTransformers
# Python user defined module
import prompts


class COD():
    '''This class implements the Chain-Of-Density summarization'''
    
    def __init__(self):
        self.llm =  CTransformers(model="TheBloke/Llama-2-13B-chat-GGUF",
                    model_file='llama-2-13b-chat.Q4_K_S.gguf',
                    model_type='llama', config={'max_new_tokens': 1024, 'threads': 8, 'context_length': 2048}
        )
    
    def __call__(self, text_content):
        try:
            start_time = time.time()
            print("[INFO] The Chain Of Density summarization started...")
            kw_extract_messages = [
                                      HumanMessage(content=prompts.KW_EXTRACT_SYSTEM_PROMPT.format(text_chunk=text_content))
                                  ]
            
            cod_messages = [
                                SystemMessage(content=prompts.COD_SYSTEM_PROMPT),
                                HumanMessage(content="Here is the input text for you to summarize using the 'Missing_Entities' and 'Denser_Summary' approach:\n\n{}".format(text_content))
                           ]
            
            with get_openai_callback() as openai_cb:
                kw_response = self.llm(kw_extract_messages)
                cod_response = self.llm(cod_messages)
            
            kw_output = kw_response.content.split(", ")
            output = cod_response.content
            
            try:
                output_dict = json.loads(output.replace("\n", ""))
                summary = output_dict[-1]['Denser_Summary']
                print("[INFO] The Chain Of Density summarization done!")
                end_time = time.time()
                return {"summary": summary,
                        "keywords": kw_output}
            except json.JSONDecodeError:
                print("[ERROR] The output JSON is not valid of the COD prompt response. LLM Output:\n\n{}\n\n".format(output))
                return
            except KeyError:
                print("[ERROR] The COD output JSON is missing the key `Denser_Summary`. Valid keys are `Missing_Entities` & `Denser_Summary`. LLM Output:\n\n{}\n\n".format(output))
                return
        except Exception as error:
            print("[ERROR] Some error happend in COD. Error:\n\n{}\n\n".format(error))
            traceback.print_exception(type(error), error, error.__traceback__)
            return