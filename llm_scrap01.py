
import traceback
from typing import Optional, Any, List, Union
from lightrag import LightRAG
import asyncio, nest_asyncio 
from lightrag.utils import logger
#from google import genai
import google.genai as genai
from google.genai import types, errors, Client

nest_asyncio.apply()  ##

# LightRAG wrapper class
class LightRAGApp:
    """LightRAG application wrapper with async support"""
    
    def __init__(self):
        """Initialise LightRAG application state"""
        self.rag: Optional[LightRAG] = None
        self.working_dir: Optional[str] = None
        self.llm_backend: Optional[str] = None
        self.llm_model_name: Optional[str] = None
        self.llm_model_embed: Optional[str] = None
        self.llm_baseurl: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.status: str = ""
        self._is_initialised: bool = False  ## Add initialisation flag
        self.cancel_event = asyncio.Event()  ## Add cancel event: long-running tasks
        self.delay_between_files: Optional[float]=60.0  ## lightRAG initialisation: Delay in seconds between files processing viz RateLimitError 429
        self.llm_model_max_async: Optional[int] = 2,  #4,  ##SMY: https://github.com/HKUDS/LightRAG/issues/128
        self.max_parallel_insert: Optional[int] = 1,  ## No of parralel files to process in one batch: aasist: https://github.com/HKUDS/LightRAG/issues/1653#issuecomment-2940593112
        self.timeout: Optional[float] = 1000,  #AsyncOpenAI #Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        self.max_retries: Optional[int] = 1  #AsyncOpenAI #DEFAULT_MAX_RETRIES, 

# Call GenAI   ##SMY: to do: Follow GenAI or map to ligthRAG's openai_complete()
    #async def genai_complete(self, prompt, system_prompt=None, history_messages: Optional[List[types.Content]] = None, **kwargs) -> Union[str, types.Content]:
    async def genai_complete(self, model: str, prompt: str, system_prompt: Union[str, None] =None, 
                             history_messages: Union[Optional[List[types.Content]], None] = None,
                             api_key: Union[str, None] = None,
                               **kwargs) -> Union[str, types.Content]:
        """ Create GenAI client and complete a prompt """
        # https://github.com/googleapis/python-genai/tree/main

        # 1. Combine prompts: system prompt, history, and user prompt
        if not history_messages or history_messages is None:
            history_messages = []

        # prepare message
        #messages: list[dict[str, Any]] = []
        messages: list[types.Content] = []
        
        '''if system_prompt:   ##See system_instruction
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})'''

        #if system_prompt:   ##See system_instruction
        #    messages.append(types.Content(role="system", parts=[system_prompt]))
        #messages.append(types.Content(role="user", parts=[history_messages]))
        #messages.append(history_messages if history_messages and isinstance(history_messages, types.Content) else types.Content(role="user", parts=history_messages))
        #messages.append(types.Content(role="user", parts=[prompt]))

        if system_prompt:   ##See system_instruction
            history_messages.append(types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]))
        new_user_content =  types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        history_messages.append(new_user_content)

        logger.debug(f"Sending messages to Gemini: Model: {self.llm_model_name.rpartition("/")[-1]} \n~ Message: {prompt}")
        logger_kg.log(level=20, msg=f"Sending messages to Gemini: Model: {self.llm_model_name.rpartition("/")[-1]} \n~ Message: {prompt}")
        
        # 2. Initialize the GenAI Client with Gemini API Key
        client = Client(api_key=self.llm_api_key)     #api_key=gemini_api_key
        #aclient = genai.Client(api_key=self.llm_api_key).aio  # use AsyncClient

        # 3. Call the Gemini model. Don't use async with context manager, use client directly.
        try:
            response = client.models.generate_content(
            #response = await aclient.models.generate_content(
                model = self.llm_model_name.rpartition("/")[-1],   #"gemini-2.0-flash",
                #contents = [combined_prompt],
                contents = history_messages,   #messages,
                config = types.GenerateContentConfig(
                    #max_output_tokens=5000, 
                    temperature=0, top_k=10, top_p=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                    system_instruction="You are an expert in Knowledge graph. You are well versed in entities, relations, objects and ontology reasoning", #system_prompt,                    
                )
            )
            logger_kg.log(level=30, msg=f"GenAI response: \n ", extra={"Model": response.text})
            #response.text
            
        except errors.APIError as e:
            logger.error(f"GenAI API error: code: {e} ~ Status: {e.status}")
            logger_kg.log(level=30, msg=f"Gen API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Status: {e.status}")
            
            client.close()  # Ensure client is closed
            #await aclient.close()  # .aclose()
            raise
        except Exception as e:
            logger.error(
                f"GenAI API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Traceback: {traceback.format_exc()}"
            )
            logger_kg.log(level=30, msg=f"GenAI API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Traceback: {traceback.format_exc()}")
            
            client.close()  # Ensure client is closed
            #await aclient.close()  # .aclose()
            raise 
    
        # 5. Return the response text
        return response.text

##### START #####
#import logging.config
#from utils.logger import get_logger, setup_logging
#logger_kg = get_logger(__name__)   ## app logging

if __name__ == "__main__":
    #gradio_ui().launch() 
    
    ##SMY: assist: https://www.gradio.app/guides/developing-faster-with-reload-mode
    ##SMY: NB: gradio app_gradio_lightrag.py --demo-name=gradio_ui
    async def main():
        #from app_gradio_lightrag import LightRAGApp
        # Instantiate LightRAG and launch Gradio
        try:
            app_logic = LightRAGApp()
            #gradio_ui(app_logic).launch(server_port=7866)

            app_logic.llm_api_key = "AIzaSyA3ES6nVb6B_NQTftqVKI8dnKn9ALsc3gM"
            app_logic.llm_model_name = "google/gemini-2.5-flash"
            prompt = "Define Knowledge graph. Do not explain"            
            
            logger_kg.log(level=20, msg="GenAI: ", extra={"model: ": app_logic.llm_model_name})

            result = await app_logic.genai_complete(prompt=prompt)

            logger_kg.log(level=20, msg="GenAI response: ", extra={"response: ": result})
        except Exception as e:
            print(f"An error occurred: {e} ~ Traceback {traceback.format_exc()}")
        finally:
            if app_logic.rag:
                await app_logic.rag.finalize_storages()
 
    from utils.logger import get_logger, setup_logging
    setup_logging()         ## set logging
    logger_kg = get_logger("semmyKG")   ## app logging

    ##SMY Initialise logging before running the main function: See lightrag_openai_compatible_demo.py
    #from app_gradio_lightrag import handle_errors, configure_logging
    #configure_logging()     ## lightRAG logging
    
    asyncio.run(main())
