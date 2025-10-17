import os   ## replace with Path
from pathlib import Path
import glob

import gradio as gr
#from watchfiles import run_process  ##gradio reload watch
import numpy as np  ##SMY
import random

from functools import partial
from typing import Tuple, Optional, Any, List, Union

import inspect  ##SMY lightrag_openai_compatible_demo.py

def install(package):
    import subprocess
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])
try:
    import pipmaster as pm
except ModuleNotFoundError:  ##assist: https://discuss.huggingface.co/t/huggingface-spaces-not-updating-packages-from-requirements-txt/92865/4?u=semmyk
    install("pipmaster")
    import pipmaster as pm
if not pm.is_installed("nest_asyncio"):
    pm.install("nest_asyncio")    #HF Spaces modulenotfounderror: No module named 'nest_asyncio'
if not pm.is_installed("google-genai"):
    pm.install("google-genai")      ## use gemini as a client: genai
if not pm.is_installed("gradio[oauth]==5.29.0"):
    pm.install("gradio[oauth]==5.29.0")
if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")
if not pm.is_installed("sentence-transformers"):
    pm.install("sentence-transformers")
if not pm.is_installed("hf_xet"):
    pm.install("hf_xet")    #HF Xet Storage downloader
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

#from google import genai
from google.genai import types, errors, Client
from openai import APIConnectionError, APIStatusError, NotFoundError, APIError, BadRequestError
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_complete, openai_embed, InvalidResponseError
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug  ##SMY
from lightrag.kg.shared_storage import initialize_pipeline_status  ##SMY

from utils.file_utils import check_create_dir, check_create_file

import asyncio
import nest_asyncio
# Apply nest_asyncio to solve event loop issues: allow nested evennt loops
nest_asyncio.apply()

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

import traceback
import logging, logging.config  ##SMY lightrag_openai_compatible_demo.py
from utils.logger import get_logger
logger_kg = get_logger(__name__)

# Pythonic error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return gr.update(value=f"Error: {e}")
    return wrapper

@handle_errors
def configure_logging():
    """Configure logging for lightRAG"""
    ##SMY lightrag_openai_compatible_demo.py

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    #log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_dir = os.getenv("LOG_DIR", "logs")
    '''log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
                )'''
    if log_dir:
        log_file_path = Path(log_dir) / "lightrag_logs.log"
    else:
        log_file_path = Path("logs") / "lightrag_logs.log"

    #log_file_path.mkdir(mode=0o2755, parents=True, exist_ok=True)
    check_create_file(log_file_path)

    #print(f"\nLightRAG log file: {log_file_path}\n")
    logger_kg.log(level=20, msg=f"LightRAG logging creation", extra={"LightRAG log file: ": log_file_path.name})

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

# Utility: Wrap async functions
##SMY: temporary dropped for async def declaration
'''def wrap_async(func):
    """Wrap an async function to run synchronously using asyncio.run"""
    async def _async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        return result
    return lambda *args, **kwargs: asyncio.run(_async_wrapper(*args, **kwargs))'''

# Utility: Visualise .graphml as HTML using pyvis
@handle_errors
def visualise_graphml(graphml_path: str, working_dir: str) -> str:
    """Convert GraphML file to interactive HTML visualisation"""
    ## graphml_path: defaults to lightRAG's generated graph_chunk_entity_relation.graphml
        ## working_dir: lightRAG's working directory set by user

    ## Load the GraphML file
    G = nx.read_graphml(graphml_path)

    ## Create a Pyvis network
    #net = Network(height="100vh", notebook=True)
    net = Network(notebook=True, width="100%", height="600px")  #, heading=f"Knowledge Graph Visualisation")  #(noteboot=False)
    ## Convert NetworkX graph to Pyvis network
    net.from_nx(G)

    # Add colors and title to nodes
    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]
    
    # Add title to edges
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    ## Set the 'physics' attribute to repulsion
    net.repulsion(node_distance=120, spring_length=200)
    net.show_buttons(filter_=['physics'])  ##SMY: dynamically modify the network
    #net.show_buttons()
    
    ## graph path
    kg_viz_html_file = "kg_viz.html"
    #html_path = os.path.join(working_dir, kg_viz_html_file)
    html_path = Path(working_dir) / kg_viz_html_file

    #net.save_graph(html_path)
    ## Save and display the generated KG network html
    #net.show(html_path)
    net.show(str(html_path), local=True, notebook=False)

    ##SMY read and display generated KG html
    #with open(html_path, "r", encoding="utf-8") as f:
    #    return f.read()  ## html

# Utility: Get all markdown files in a folder
def get_markdown_files(folder: str) -> list[str]:
    """Get sorted list of markdown files in folder"""
    #return sorted(glob.glob(os.path.join(folder, "*.md")))
    #return sorted(Path(folder).glob("*.md"))    ## change to Pathlib. SMY: We're not interested in sub-directory, hence not rglob()

    #markdown_files = sorted([file for file in Path(folder).glob("*.md")])
    markdown_files_list = sorted(str(file) for file in Path(folder).iterdir() if file.suffix == ".md")
    return markdown_files_list


# LightRAG wrapper class
class LightRAGApp:
    """LightRAG application wrapper with async support"""
    
    def __init__(self):
        """Initialise LightRAG application state"""
        self.rag: Optional[LightRAG] = None
        self.working_dir: Optional[str] = None
        self.llm_backend: Optional[str] = None
        self.embed_backend: Optional[str] = None
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

    def _system_prompt(self, custom_system_prompt: Optional[str]=None) -> str:
        """Set a localised system prompt"""
        ## SMY: TODO: Make modular
        #self.system_prompt if custom_system_prompt else self.system_prompt=f"\n 
        
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        else:
            self.system_prompt = """
            You are a domain expert on Cybersecurity, the South Africa landscape and South African legislation. 
            1. You only process text in English. You disregard pages containing other languages or typesetting not English in context. 
               - South African legislation are written in English or in English and another South African language.
            2. When building knowledge graph, taxonomy and ontology, 
               - take cognisance of NER (Named Entity Recognition) with localisation and domain-context in mind.
               - So, person(s) can be natural or juristic person. Natural person(s) are individuals, while juristic person(s) are organisations. 
               - For instance, Minister of Justice is juristic. Likewise, Information Regulator (South Africa) is juristic, while Advocate (Adv) Pansy Tlakula is natural. Ditto, Public Protector is juristic  
            3. Different natural and juristic person(s) are assigned to perform roles.
            4. In South Africa, there are different entities (organisations or departments) defined in legislations, Acts, Bills and Policy. 
               - For instance, you might have aDept of Treasury at National (The National Treasury) and at Provincial levels (Provincial Treasuries) guided by the PFMA, while
               - Municipalities (local governments), guided by the MFMA, do not have Treasury department, but might have Budget & Treasury Office.
               - You have stand alone entities like the Office of the Public Protector, headed by the Public Protector. Ditto, Information Regulator headed by Chairperson of the Information Regulator.
               - You have others like the CCMA (Commission for Conciliation, Mediation and Adjudication) that are creation of satutes.
            5. Legislations include Acts, Bill and in some instance, Regulations and Policies.
            6. Legislations often have section heads. The also have section detailing amendments and repeals (if any).
            7. Legislations will indicate the heading in the format Name Act No of YYYY. For instance 'Protection of Information Act No 84, 1982.
               - Legislation might have other Act No of YYYY as they are amended. Take cognisance and tightly keep/link to the root legislation.
               - For instance for the LRA Act, the root is Labour Relations Act 66 of 1995, while soome of the amendments are Labour Relations Amendment Act 6 of 2014, Labour Relations Amendment Act 8 of 2018
            8. Legislations will have a Gazette No and Assented date (when the President assent to the legislation) from when it becomes operative: that is ... with effect from or wef dd mmm YYYY. 
               - Certain part of a legislation might not be operative on the date the legislation is assented to.
            9. Legislation might have paragraph number. Kindly disregard for content purposes but take cognisance for context.
            10. Do not create multiple nodes for legislations, written in different formats. 
                - For instance, maintain a single node for Protection of Information Act, Protection of Information Act, 1982, Protection of Information Act No 84, 1982.
                - However, have a separate node for Protection of Personal Information Act, 2013; as it it a separate legislation.
                - Also take note that 'Republic of South Africa' is an offical geo entity while 'South Africa' is a referred to place, although also a geo entity: 
                - Always watch the context and becareful of lumping them together.
                """

        return self.system_prompt

    async def _embedding_func(self, texts: list[str], **kwargs,) -> np.ndarray:
    #def _embedding_func(self, texts: list[str], **kwargs,) -> np.ndarray:
        """Get embedding function based on backend"""
        try:
            # Use HF embedding
            if self.embed_backend == "Transformer" or self.embed_backend[0] == "Transformer" :
                model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)   #("all-MiniLM-L6-v2")
                embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                return embeddings
            # Use OpenAI
            elif self.llm_backend == "OpenAI":
                # Use wrap_async for proper async handling
                #return wrap_async(openai_embed)(
                return await openai_embed(
                    texts, 
                    model=self.llm_model_embed,                    
                    api_key=self.llm_api_key_embed,
                    base_url=self.llm_baseurl_embed
                    #client_configs=None  #: dict[str, Any] | None = None,
                )
            # Use Ollama
            elif self.llm_backend == "Ollama":
                #return wrap_async(ollama_embed)(
                return await ollama_embed(
                    texts, 
                    embed_model=self.llm_model_embed,
                    #host=self.openai_baseurl_embed
                    host=self.ollama_host,
                    api_key=self.llm_api_key_embed
                    )
            
        except Exception as e:
            self.status = f"{self.status} | _embedding_func error: {str(e)}"
            logger_kg.log(level=30, msg=f"{self.status} | _embedding_func error: {str(e)}")
            raise  # Re-raise to be caught by the setup method

    async def _get_embedding_dim(self) -> int:
    #def _get_embedding_dim(self) -> int:
        """Dynamically determine embedding dimension or fallback to defaults"""
        try:
            test_text = ["This is a test sentence for embedding."]
            
            embedding = await self._embedding_func(test_text)
            ##SMY: getting asyncio error with wrap_async
            #embedding = wrap_async(self._embedding_func)(test_text)

            return embedding.shape[1]
        except Exception as e:
            self.status = f"_get_embedding_dim error: {str(e)}"
            logger_kg.log(level=30, msg=f"_get_embedding_dim error: {str(e)}")
            # Fallback to known dimensions
            if "bge-m3" in self.llm_model_embed:
                return 1024  # BAAI/bge-m3 embedding
            if self.llm_backend == "OPENAI" and "gemini" in self.llm_model_name:
                return 3072  # Gemini's gemini-embedding-exp-03-07 dimension
            if self.llm_backend == "OpenAI":
                return 1536  # OpenAI's text-embedding-3-small            
            return 4096  # Ollama's default

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

        if system_prompt:   ##See system_instruction
            history_messages.append(types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)]))
        new_user_content =  types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        history_messages.append(new_user_content)

        logger.debug(f"Sending messages to Gemini: Model: {self.llm_model_name.rpartition('/')[-1]} \n~ Message: {prompt}")
        logger_kg.log(level=20, msg=f"Sending messages to Gemini: Model: {self.llm_model_name.rpartition('/')[-1]} \n~ Message: {prompt}")
        
        # 2. Initialize the GenAI Client with Gemini API Key
        client = Client(api_key=self.llm_api_key)     #api_key=gemini_api_key
        #aclient = genai.Client(api_key=self.llm_api_key).aio  # use AsyncClient

        # 3. Call the Gemini model. Don't use async with context manager, use client directly.
        try:
            response = client.models.generate_content(
            #response = await aclient.models.generate_content(
                model = self.llm_model_name.rpartition("/")[-1] if self.llm_model_name else "gemini-2.0-flash-exp:free",   #"gemini-2.0-flash",
                #contents = [combined_prompt],
                contents = history_messages,   #messages,
                config = types.GenerateContentConfig(
                    #max_output_tokens=5000, 
                    temperature=0, top_k=10, top_p=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                    #automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
                    system_instruction=["You are an expert in Knowledge graph.",
                                        "You are well versed in entities, relations, objects and ontology reasoning",
                                        "Your mission/task is to create/construct knowledge Graph"], #system_prompt,                    
                )
            )
            ## GenAI keeps giving pydantic error relating to 'role': 'assistant'  #wierd
            ## SMY: suspect is lightRAG prompts' examples.

            logger_kg.log(level=30, msg=f"GenAI response: \n ", extra={"Model": response.text})
            #return response.text
            
        except errors.APIError as e:
            logger.error(f"GenAI API error: code: {e} ~ Status: {e.status}")
            logger_kg.log(level=30, msg=f"Gen API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Status: {e.status}")
            
            #client.close()  # Ensure client is closed    #Err in 1.43.0
            #await aclient.close()  # .aclose()
            raise
        except Exception as e:
            logger.error(
                f"GenAI API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Traceback: {traceback.format_exc()}"
            )
            logger_kg.log(level=30, msg=f"GenAI API Call Failed,\nModel: {self.llm_model_name}\nGot: code: {e} ~ Traceback: {traceback.format_exc()}")
            
            #client.close()  # Ensure client is closed    #Err in 1.43.0
            #await aclient.close()  # .aclose()
            raise 
    
        # 4. Return the response text
        return response.text
    
    #def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False,
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        """Complete a prompt using OpenAI's API with or without caching support."""
        try:
            ## SMY: TODO: Revisit to make modular: tie-in with Gradio UI
            if not system_prompt:
                system_prompt = self._system_prompt()
        except Exception as e:
            self.status = f"_llm_model_func: Error while setting system_promt: {str(e)}"
            logger_kg.log(level=30, msg=f"_llm_model_func: Error while setting system_promt: {str(e)}")
            raise
        
                    
        self.status = f"{self.status}\n _llm_model_func: calling LLM to process ... with {self.llm_backend}"
        logger_kg.log(level=20, msg=f"{self.status}\n _llm_model_func: calling LLM to process ... with {self.llm_backend}")

        try:
            await asyncio.sleep(self.delay_between_files/6)  # Pause between file processing  #10s
            if self.llm_backend == "GenAI":
                return await self.genai_complete(
                    model=self.llm_model_name, 
                    prompt=prompt, 
                    system_prompt=system_prompt, 
                    history_messages=history_messages,
                    #base_url=self.llm_baseurl, 
                    api_key=self.llm_api_key,
                    **kwargs
                )
            #elif self.llm_backend == "OpenAI":
            else:
                #return openai_complete_if_cache(
                return await openai_complete_if_cache(
                    model=self.llm_model_name.rpartition('/')[-1] if "googleapi" in self.llm_baseurl else self.llm_model_name,  #"gemini" in self.llm_model_name else self.llm_model_name, 
                    prompt=prompt, 
                    system_prompt=system_prompt, 
                    history_messages=history_messages,
                    base_url=self.llm_baseurl, 
                    api_key=self.llm_api_key, 
                    #timeout=self.timeout,  #: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
                    #max_retries=self.max_retries,  #: int = DEFAULT_MAX_RETRIES,
                    **kwargs,
                )
        except Exception as e:
            self.status = f"_llm_model_func: Error while initialising model: {str(e)}"
            logger_kg.log(level=30, msg=f"_llm_model_func: Error while initialising model: {str(e)}")
            raise

    def _ensure_working_dir(self) -> str:
        """Ensure working directory exists and return status message"""
        '''if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            return f"Created working directory: {self.working_dir}"'''
        if not Path(self.working_dir).exists():
            check_create_dir(self.working_dir)
            return f"Created working directory: {self.working_dir}"
        return f"Working directory exists: {self.working_dir}"


    async def _initialise_storages(self) -> str:
    #def _initialise_storages(self) -> str:
        """Initialise LightRAG storages and pipeline"""
        try:
            #wrap_async(self.rag.initialize_storages)
            #wrap_async(initialize_pipeline_status)
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            return "Storages and pipeline initialised successfully"
        except Exception as e:
            return f"Storage initialisation failed: {str(e)}"

    ##SMY: 
    async def _initialise_rag(self):
        """Initialise lightRAG"""

        ##debug
        # ## getting embedidngs dynamically
        #self.status = f"Getting embeddings dynamically"
        #print(f"Getting embeddings dynamically")
        #print(f"_embedding_func: llm_model_embed: {self.llm_model_embed}")
        #print(f"_embedding_func: llm_api_key_embed: {self.llm_api_key_embed}")
        #print(f"_embedding_func: llm_baseurl_embed: {self.llm_baseurl_embed}")
        
        # Clear old data files
        #wrap_async(self._clear_old_data_files)
        #await self._clear_old_data_files()
        """Clear old data files"""
        files_to_delete = [
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ]
        
        for file in files_to_delete:
            '''file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")'''
            file_path = Path(self.working_dir) / file
            if file_path.exists():
                file_path.unlink()
                logger_kg.log(level=20, msg=f"LightRAG class: Deleting old files", extra={"filepath": file_path.name})

        
        # Get embedding
        if self.embed_backend == "Transformer" or self.embed_backend[0] == "Transformer":
            logger_kg.log(level=20, msg=f"Getting embeddings dynamically through _embedding_func: ", 
                          extra={"embedding backend": self.embed_backend, })
        else:
            logger_kg.log(level=20, msg=f"Getting embeddings dynamically through _embedding_func: ", extra={
                "embedding backend": self.embed_backend,
                "llm_model_embed": self.llm_model_embed,
                "llm_api_key_embed": self.llm_api_key_embed,
                "llm_baseurl_embed": self.llm_baseurl_embed,
            })
        #embedding_dimension = wrap_async(self._get_embedding_dim)
        embedding_dimension = await self._get_embedding_dim()
        
        #print(f"Detected embedding dimension: {embedding_dimension}")
        logger_kg.log(level=20, msg=f"Detected embedding dimension: ", extra={"embedding_dimension": embedding_dimension, "embedding_type": self.embed_backend})

        try:
            rag = LightRAG(
                working_dir=self.working_dir,
                #llm_model_max_async=self.llm_model_max_async,  #getting tuple instead of int   #1, #opting for lightRAG default
                #max_parallel_insert=self.max_parallel_insert,  #getting tuple instead of int   #1,  #opting for lightRAG default
                llm_model_name=self.llm_model_name.rpartition("/")[-1],   #self.llm_model_name,
                llm_model_func=self._llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),    #8192,
                    func=self._embedding_func,
                ),
            )
            self.rag = rag

           # Initialise RAG instance
            #wrap_async(self._initialise_storages)
            await self._initialise_storages()
            
            #await rag.initialize_storages()
            #await initialize_pipeline_status()  ##SMY: still relevant in updated lightRAG? - """Asynchronously finalize the storages"""

            self.status = f"Storages and pipeline initialised successfully"  ##SMY: debug
            logger_kg.log(level=20, msg=f"Storages and pipeline initialised successfully")
            return self.rag        #return rag
        except Exception as e:
            tb = traceback.print_exc()
            return f"lightRAG initialisation failed: {str(e)} \n traceback: {tb}"
            #raise RuntimeWarning(f"lightRAG initialisation failed: {str(e.with_traceback())}")

    @handle_errors
    #def setup(self, data_folder: str, working_dir: str, llm_backend: str,
    async def setup(self, data_folder: str, working_dir: str, llm_backend: str, embed_backend: str,
             openai_key: str, openai_baseurl: str, openai_baseurl_embed: str, llm_model_name: str, 
             llm_model_embed: str, ollama_host: str, embed_key: str) -> str:
        """Set up LightRAG with specified configuration"""
        # Configure environment
        #os.environ["OPENAI_API_KEY"] = openai_key or os.getenv("OPENAI_API_KEY", "")
        ##os.environ["OLLAMA_HOST"] = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        #os.environ["OLLAMA_API_BASE"] = os.getenv("OLLAMA_API_BASE")  #, "http://localhost:1337/v1/chat/completions")
        ##os.environ["OPENAI_API_BASE"] = openai_baseurl or os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        #os.environ["OPENAI_API_EMBED_BASE"] = openai_baseurl_embed or os.getenv("OPENAI_API_EMBED_BASE")  #, "http://localhost:1234/v1/embeddings")

        # Update instance state
        self.data_folder = data_folder
        self.working_dir = working_dir
        self.llm_backend = llm_backend
        self.embed_backend = embed_backend if isinstance(embed_backend, str) else embed_backend[0],
        self.llm_model_name = llm_model_name
        self.llm_model_embed = llm_model_embed
        self.llm_baseurl = openai_baseurl
        self.llm_baseurl_embed = openai_baseurl_embed
        self.llm_api_key = openai_key
        self.ollama_host = ollama_host
        self.llm_api_key_embed = embed_key
        
        try:
            ## ensure working folder exists and send status
            try:
                self.status = self._ensure_working_dir()
            except Exception as e:
                self.status = f"LightRAG initialisation.setup: working dir err | {str(e)}"

            # Initialize lightRAG with storages
            try:
                #self.rag = wrap_async( self._initialise_rag)
                self.rag = await self._initialise_rag()
                self.status = f"{self.status}\n{self.rag}"

                # set LightRAG class initialised flag
                self._is_initialised = True
                self.status = f"{self.status}\n Initialised LightRAG with {llm_backend} backend"   
                logger_kg.log(level=20, msg=f"{self.status}\n Initialised LightRAG with {llm_backend} backend" )
            except Exception as e:
                tb = traceback.print_exc()
                self.status = f"{self.status}\n LightRAG initialisation.setup and storage failed | {str(e)}"
                logger_kg.log(level=30, msg=f"{self.status}\n LightRAG initialisation.setup and storage failed | {str(e)} \n traceback: {tb}")
            
        except Exception as e:
            self._is_initialised = False
            tb = traceback.format_exc()
            self.status = (f"LightRAG initialisation failed: {str(e)}\n"
                         f"LightRAG with {working_dir} and {llm_backend} not initialised")
            logger_kg.log(level=30, msg=f"LightRAG with {working_dir} and {llm_backend} not initialised"
                         f"LightRAG initialisation failed: {str(e)}\n{tb}")
            return self.status
            
        return self.status
    
    ''' ##SMY: disabled to follow lightRAG documentations
    @handle_errors
    #def setup(self, data_folder: str, working_dir: str, llm_backend: str,
    async def setup(self, data_folder: str, working_dir: str, llm_backend: str, 
             openai_key: str, llm_baseurl: str, llm_model_name: str, 
             llm_model_embed: str) -> str:
        """Set up LightRAG with specified configuration"""    
    '''

    @handle_errors
    async def index_documents(self, data_folder: str) -> Tuple[str, str]:
    #def index_documents(self, data_folder: str) -> Tuple[str, str]:
        """Index markdown documents with progress tracking"""
        if not self._is_initialised or self.rag is None:
            return "Please initialise LightRAG first using the 'Initialise App' button.", "Not started"
            
        md_files = get_markdown_files(data_folder)
        if not md_files:
            return f"No markdown files found in {data_folder}:", "No files"
            
        try:
            total_files = len(md_files)
            #self.status = f"Starting to index {total_files} files..."
            status_msg = f"Starting to index {total_files} files"
            progress_msg = f"Found {total_files} files to index"
            
            self.reset_cancel()  ## Add <-- Reset at the start of each operation. ##TODO: ditto for query

            ##SMY: opted for enumerated lightRAG ainsert to handle LLM RateLimitError 429
            for idx, md_file in enumerate(md_files, 1):
                ## cancel indexing
                if self.cancel_event.is_set():
                    self.status = "Indexing cancelled by user."
                    logger_kg.log(level=20, msg=f"{self.status}")
                    return self.status, "Cancelled"
                else:
                    #delay_between_files: float=60.0  ## Delay in seconds between files processing viz RateLimitError 429
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            text = f.read()
                            ##SMY: 15Oct25. Prefix text with 'Search_document' to aid embedding indexing
                            text = "Search_document" + text

                        #status_msg = f"Indexing file {idx}/{total_files}: {os.path.basename(md_file)}"
                        #progress_msg = f"Processing {idx}/{total_files}: {os.path.basename(md_file)}"
                        status_msg = f"Indexing file {idx}/{total_files}: {Path(md_file).name}"
                        progress_msg = f"Processing {idx}/{total_files}: {Path(md_file).name}"
                        
                        # Use wrap_async for proper async handling
                        ###wrap_async(self.rag.)(text, file_paths=md_file)
                        await self.rag.ainsert(text, file_paths=md_file)  ##SMY: TODO [12Oct25]: Err: "object of type 'WindowsPath' has no len()"
                        #wrap_async(self.rag.ainsert)(input=text, filepaths=md_file)

                        await asyncio.sleep(self.delay_between_files)  # Pause between file processing
                        
                        status_msg = f"{self.status}\n Successfully indexed {total_files} markdown files."
                        progress_msg = f"{self.status}\n Completed: {total_files} files indexed"
                        logger_kg.log(level=20, msg=f"{self.status}\n Successfully indexed {total_files} markdown files.")
                    
                    #'''   ##SMY: flagged: to delete
                    except (NotFoundError, InvalidResponseError, APIError, APIStatusError, APIConnectionError, BadRequestError):    ##limit_async
                        # Get model name excluding the model provider
                        self.rag.llm_model_name = self.llm_model_name.rpartition("/")[-1]
                        status_msg = f"Retrying indexing file {idx}/{total_files}: {Path(md_file).name}"
                        progress_msg = f"Retrying processing {idx}/{total_files}: {Path(md_file).name}"
                        
                        # Use wrap_async for proper async handling
                        ###wrap_async(self.rag.)(text, file_paths=md_file)
                        await self.rag.ainsert(text, file_paths=md_file)  ##SMY: TODO [12Oct25]: Err: "object of type 'WindowsPath' has no len()"
                        #wrap_async(self.rag.ainsert)(input=text, filepaths=md_file)

                        await asyncio.sleep(self.delay_between_files)  # Pause between file processing
                    #'''
                    except Exception as e:
                        tb = traceback.print_exc()
                        #self.status = f"Error indexing {os.path.basename(md_file)}: {str(e)}"
                        status_msg = f"Error indexing {Path(md_file).name}: {str(e)}"
                        progress_msg = f"Failed on {idx}/{total_files}: {Path(md_file).name}"
                        logger_kg.log(level=30, msg=f"Error indexing: Failed on {idx}/{total_files}: {Path(md_file).name} - {str(e)} \n traceback: {tb}")
                        continue
                await asyncio.sleep(1)  #(0) ## Add Yield to event loop
                    
        except Exception as e:
            tb = traceback.print_exc()
            status_msg = f"{self.status}\n Indexing failed: {str(e)}"
            progress_msg = "{self.status}\n Indexing failed"
            logger_kg.log(level=30, msg=f"{self.status}\n Indexing failed: {str(e)} \n traceback: {tb}")
            
        '''status_msg = f"{self.status}\n Successfully indexed {total_files} markdown files."
            progress_msg = f"{self.status}\n Completed: {total_files} files indexed"
            logger_kg.log(level=20, msg=f"{self.status}\n Successfully indexed {total_files} markdown files.")'''
        
        return status_msg, progress_msg

    @handle_errors
    async def query(self, query_text: str, mode: str) -> str:
    #def query(self, query_text: str, mode: str) -> str:
        """Query LightRAG with specified mode"""
        if not self._is_initialised or self.rag is None:
            return (f"Please initialise LightRAG first using the 'Initialise App' button. \n"
                    f" and index with 'Index Documents' button")
            
        param = QueryParam(mode=mode)
        
        ## return lightRAG query answer
        # Use wrap_async for proper async handling
        ###return await wrap_async(self.rag.aquery)(query_text, param=param)
        #return wrap_async(self.rag.aquery)(query_text, param=param)
        return await self.rag.aquery(query_text, param=param)  ##SMY: 
    
        #####Err
        ##return lambda *args, **kwargs: asyncio.run(_async_wrapper(*args, **kwargs))
        ##File "C:\Dat\dev\Python\Python312\Lib\asyncio\runners.py", line 190, in run
        ##raise RuntimeError(
        ##RuntimeError: asyncio.run() cannot be called from a running event loop 

    @handle_errors
    def show_kg(self) -> str:
        """Display knowledge graph visualisation"""
        ## graphml_path: defaults to lightRAG's generated graph_chunk_entity_relation.graphml
        ## working_dir: lightRAG's working directory set by user  
        '''graphml_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_path):
            return "Knowledge graph file not found. Please index documents first to generate Knowledge Graph."'''
        graphml_path = Path(self.working_dir) / "graph_chunk_entity_relation.graphml"
        if not Path(graphml_path).exists():
            return "Knowledge graph file not found. Please index documents first to generate Knowledge Graph."
        
        #return visualise_graphml(graphml_path)
        return visualise_graphml(graphml_path, self.working_dir)

    def reset_cancel(self):
        """Reset cancel event"""
        self.cancel_event.clear()

    def trigger_cancel(self):
        """Set cancel event"""
        self.cancel_event.set()


############
'''    
    ##SMY: //TODO: Gradio toggle button
    def _clear_old_data_files(self):
        """Clear old data files"""
        files_to_delete = [
                    "graph_chunk_entity_relation.graphml",
                    "kv_store_doc_status.json",
                    "kv_store_full_docs.json",
                    "kv_store_text_chunks.json",
                    "vdb_chunks.json",
                    "vdb_entities.json",
                    "vdb_relationships.json",
                ]
        
        for file in files_to_delete:
            file_path = Path(self.working_dir) / file
            if file_path.exists():
                file_path.unlink()
                logger_kg.log(level=20, msg=f"LightRAG class: Deleting old files", extra={"filepath": file_path.name})'''
'''

    async def _get_llm_functions(self) -> Tuple[callable, callable]:
    #def _get_llm_functions(self) -> Tuple[callable, callable]:
        """Get LLM and embedding functions based on backend"""
        try:
            # Get embedding dimension dynamically
            try:
                embedding_dimension = await self._get_embedding_dim()
                self.status = f"Using embedding dimension: {embedding_dimension}"
                logger_kg.log(level=20, msg=f"Using embedding dimension: {embedding_dimension}")
            except Exception as e:                
                # feedback dimensions error                
                self.status = f"_get_llm_function: embedding_dim error with fallback: {str(e)}"

            # Create embedding function wrapper: # Wrap with EmbeddingFunc to provide required attributes
            embed_func = EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=8192,  #4096,  #8192,  # Conservative default | #ollama
                func=self._embedding_func
            )
            
            # Get LLM function
            #llm_func = await self._llm_model_func  ##SMY: not used
            
            # return LLM and embed functions
            #return llm_func, embed_func
            return await self._llm_model_func(), embed_func
            
        except Exception as e:
            self.status = f"{self.status} \n| _get_llm_functions error: {str(e)}"
            logger_kg.log(level=30, msg=f"{self.status} \n| _get_llm_functions error: {str(e)}")
            raise  # Re-raise to be caught by the setup method
    '''

'''
    ##SMY: record only. for deletion
                # Wrap with EmbeddingFunc to provide required attributes
                embed_func = EmbeddingFunc(
                    #embedding_dim=1536,  # OpenAI's text-embedding-3-small dimension
                    #max_token_size=8192,  # OpenAI's max token size
                    embedding_dim=3072,  # Gemini's gemini-embedding-exp-03-07 dimension
                    max_token_size=8000,  # Gemini's embedding max token size = 20000
                    func=embedding_func
                )    
    '''

# Instantiate app logic
#app_logic = LightRAGApp()  ##SMY: already instantiated in app.main()

# Gradio UI  ## moved to app.py
#def gradio_ui():
# ...
#     return gradio_ui

#if __name__ == "__main__":
    #gradio_ui().launch() 
# ...
