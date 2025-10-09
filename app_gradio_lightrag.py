import os
import glob
import gradio as gr
from watchfiles import run_process  ##gradio reload watch

import pipmaster as pm
if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")
import networkx as nx
from pyvis.network import Network
import random

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_complete, openai_embed
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug  ##SMY
from lightrag.kg.shared_storage import initialize_pipeline_status  ##SMY

import numpy as np  ##SMY

import asyncio
from functools import partial
from typing import Tuple, Optional
import logging, logging.config  ##SMY lightrag_openai_compatible_demo.py
import inspect  ##SMY lightrag_openai_compatible_demo.py

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

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
    """Configure logging for the application"""
    ##SMY lightrag_openai_compatible_demo.py

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

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
def wrap_async(func):
    """Wrap an async function to run synchronously using asyncio.run"""
    async def _async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        return result
    return lambda *args, **kwargs: asyncio.run(_async_wrapper(*args, **kwargs))

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
    html_path = os.path.join(working_dir, kg_viz_html_file)
    #net.save_graph(html_path)
    ## Save and display the generated KG network html
    #net.show(html_path)
    net.show(html_path, local=True, notebook=False)

    ##SMY read and display generated KG html
    #with open(html_path, "r", encoding="utf-8") as f:
    #    return f.read()  ## html


# Utility: Get all markdown files in a folder
def get_markdown_files(folder: str) -> list[str]:
    """Get sorted list of markdown files in folder"""
    return sorted(glob.glob(os.path.join(folder, "*.md")))

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

    def _system_prompt(self, custom_system_prompt: Optional[str]=None) -> str:
        """Set a localised system prompt"""
        ## SMY: TODO: Make modular
        #self.system_prompt if custom_system_prompt else self.system_prompt=f"\n 
        
        if custom_system_prompt:
            self.system_prompt = custom_system_prompt
        else:
            self.system_prompt = """
            You are a domain expert on Cybersecurity, the South Africa landscape
            and South African legislation. 
            1. You only process text in English. 
            2. When building knowledge graph, taxonomy and ontology, 
            person(s) can be natural or juristic person. For instance, Minister of Justice is juristic.
            3. Different natural and juristic person(s) are assigned to perform roles.
            4. In South Africa, there are different entities (organisations) defined in legislations, Acts, Bills and Policy. 
               For instance, you might Dept of Treasury at National (The National Treasury) and at Provincial levels (Provincial Treasuries) guided by the PFMA, while
               Municipalities (local governments), guided by the MFMA, do not have Treasury department, but might have Budget & Treasury Office.
               You have stand alone entities like the Office of the Public Protector, headed by the Public Protector. Ditto, Information Regulator headed by Chairperson of the Information Regulator.
               You have others like the CCMA (Commission for Conciliation, Mediation and Adjudication)
            5. Legislations include Acts, Bill and in some instance, Regulations and Policy.
            6. Legislations often have section heads. The also have section detailing amendments and repeals (if any).
            7. Legislations will indicate the heading in the format Name Act No of YYYY. For instance 'Protection of Information Act No 84, 1982.
            8. Legislations will have a Gazette No and Assented date (when the President assent to the legislation) from when it becomes operative.
            9. Legislation might have paragraph number. Kindly disregard for content purposes but take cognisance for context.
            10. Do not create multiple nodes for legislations. For instance, maintain a single node for Protection of Information Act, Protection of Information Act, 1982, Protection of Information Act No 84, 1982.
                However, have a separate node for Protection of Personal Information Act, 2013.
                Also take note that 'Republic of South Africa' is an offical geo entity while 'South Africa' is a referred to place, although also a geo entity: Always watch the context and becareful of lumping them together.
                """

        return self.system_prompt

    async def _embedding_func(self, texts: list[str], **kwargs,) -> np.ndarray:
    #def _embedding_func(self, texts: list[str], **kwargs,) -> np.ndarray:
        """Get embedding function based on backend"""
        try:
            if self.llm_backend == "OpenAI":
                #'''
                
                # Use wrap_async for proper async handling
                #return wrap_async(openai_embed)(
                return await openai_embed(
                    texts, 
                    model=self.llm_model_embed,                    
                    api_key=self.llm_api_key_embed,
                    base_url=self.llm_baseurl_embed
                    #base_url=self.ollama_host
                )
            # Use wrap_async for proper async handling
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
            raise  # Re-raise to be caught by the setup method

    async def _get_embedding_dim(self) -> int:
    #def _get_embedding_dim(self) -> int:
        """Dynamically determine embedding dimension or fallback to defaults"""
        try:
            test_text = ["This is a test sentence."]
            embedding = await self._embedding_func(test_text)
            ##SMY: getting asyncio error
            #embedding = wrap_async(self._embedding_func)(test_text)
            return embedding.shape[1]
        except Exception as e:
            self.status = f"_get_embedding_dim error: {str(e)}"
            # Fallback to known dimensions
            if "bge-m3" in self.llm_model_embed:
                return 1024  # BAAI/bge-m3 embedding
            if self.llm_backend == "OPENAI" and "gemini" in self.llm_model_name:
                return 3072  # Gemini's gemini-embedding-exp-03-07 dimension
            if self.llm_backend == "OpenAI":
                return 1536  # OpenAI's text-embedding-3-small            
            return 4096  # Ollama's default

    #def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False,
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
        """Complete a prompt using OpenAI's API with or without caching support."""
        try:
            ## SMY: TODO: Revisit to make modular: tie-in with Gradio UI
            if not system_prompt:
                system_prompt = self._system_prompt()
        except Exception as e:
            self.status = f"_llm_model_func: Error while setting system_promt: {str(e)}"
            raise
        try:
            #return openai_complete_if_cache(
            return await openai_complete_if_cache(
                model=self.llm_model_name, 
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
            raise

    async def _get_llm_functions(self) -> Tuple[callable, callable]:
    #def _get_llm_functions(self) -> Tuple[callable, callable]:
        """Get LLM and embedding functions based on backend"""
        try:
            # Get embedding dimension dynamically
            try:
                embedding_dimension = await self._get_embedding_dim()
                self.status = f"Using embedding dimension: {embedding_dimension}"
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
            raise  # Re-raise to be caught by the setup method
    
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

    def _ensure_working_dir(self) -> str:
        """Ensure working directory exists and return status message"""
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
            return f"Created working directory: {self.working_dir}"
        return f"Working directory exists: {self.working_dir}"
    
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
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

    async def _initialise_storages(self) -> str:
    #def _initialise_storages(self) -> str:
        """Initialise LightRAG storages and pipeline"""
        try:
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
        print(f"Getting embeddings dynamically")
        print(f"_embedding_func: llm_model_embed: {self.llm_model_embed}")
        print(f"_embedding_func: llm_api_key_embed: {self.llm_api_key_embed}")
        print(f"_embedding_func: llm_baseurl_embed: {self.llm_baseurl_embed}")
        # Get embedding
        embedding_dimension = await self._get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        try:
            rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_max_async=self.llm_model_max_async,  #1,  #4,  ##SMY: https://github.com/HKUDS/LightRAG/issues/128
                max_parallel_insert=self.max_parallel_insert,  #1,  ## No of parralel files to process in one batch: assist: https://github.com/HKUDS/LightRAG/issues/1653#issuecomment-2940593112
                llm_model_func=self._llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=8192,
                    func=self._embedding_func,
                ),
            )

            await rag.initialize_storages()
            await initialize_pipeline_status()

            self.status = f"Storages and pipeline initialised successfully"  ##SMY: debug
            return rag        
        except Exception as e:
            return f"lightRAG initialisation failed: {str(e)}"

    @handle_errors
    #def setup(self, data_folder: str, working_dir: str, llm_backend: str,
    async def setup(self, data_folder: str, working_dir: str, llm_backend: str, 
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
                self.rag = await self._initialise_rag()
                self.status = f"{self.status}\n{self.rag}"

                # set LightRAG class initialised flag
                self._is_initialised = True
                self.status = f"{self.status}\n Initialised LightRAG with {llm_backend} backend"   
            except Exception as e:
                self.status = f"{self.status}\n LightRAG initialisation.setup and storage failed | {str(e)}"
            
        except Exception as e:
            self._is_initialised = False
            self.status = (f"LightRAG initialisation failed: {str(e)}\n"
                         f"LightRAG with {working_dir} and {llm_backend} not initialised")
            
        return self.status
    
    ''' ##SMY: disable to follow lightRAG documentations
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
            for idx, md_file in enumerate(md_files, 1):
                ## cancel indexing
                if self.cancel_event.is_set():
                    self.status = "Indexing cancelled by user."
                    return self.status, "Cancelled"
                else:
                    #delay_between_files: float=60.0  ## Delay in seconds between files processing viz RateLimitError 429
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            text = f.read()
                        status_msg = f"Indexing file {idx}/{total_files}: {os.path.basename(md_file)}"
                        progress_msg = f"Processing {idx}/{total_files}: {os.path.basename(md_file)}"
                        # Use wrap_async for proper async handling
                        #wrap_async(self.rag.ainsert)(text, file_paths=md_file)
                        await self.rag.ainsert(text, file_paths=md_file)  ##SMY: 
                        await asyncio.sleep(self.delay_between_files)  # Pause between file processing
                    except Exception as e:
                        #self.status = f"Error indexing {os.path.basename(md_file)}: {str(e)}"
                        status_msg = f"Error indexing {os.path.basename(md_file)}: {str(e)}"
                        progress_msg = f"Failed on {idx}/{total_files}: {os.path.basename(md_file)}"
                        continue
                await asyncio.sleep(1)  #(0) ## Add Yield to event loop
                    
            status_msg = f"{self.status}\n Successfully indexed {total_files} markdown files."
            progress_msg = f"{self.status}\n Completed: {total_files} files indexed"
        except Exception as e:
            status_msg = f"{self.status}\n Indexing failed: {str(e)}"
            progress_msg = "{self.status}\n Indexing failed"
            
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
        #return await wrap_async(self.rag.aquery)(query_text, param=param)
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
        graphml_path = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_path):
            return "Knowledge graph file not found. Please index documents first to generate Knowledge Graph."
        #return visualise_graphml(graphml_path)
        return visualise_graphml(graphml_path, self.working_dir)

    def reset_cancel(self):
        """Reset cancel event"""
        self.cancel_event.clear()

    def trigger_cancel(self):
        """Set cancel event"""
        self.cancel_event.set()

# Instantiate app logic
app_logic = LightRAGApp()

# Gradio UI
def gradio_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="LightRAG Knowledge Graph App") as gradio_ui: #demo:
        gr.Markdown("""
        # LightRAG-based Knowledge Graph RAG
        Upload your markdown docs, index and build a knowledge graph, and query with OpenAI or Ollama. Visualise the KG interactively.
        """)
        with gr.Row():
            data_folder = gr.Textbox(value="dataset/data/docs", label="Data Folder (markdown only)")
            working_dir = gr.Textbox(value="./working_folder", label="lightRAG working folder")
            llm_backend = gr.Radio(["OpenAI", "Ollama"], value="OpenAI", label="LLM Backend: OpenAI or Local")
            llm_model_name = gr.Textbox(value=os.getenv("LLM_MODEL", ""), label="LLM Model Name")  #.split('/')[1], label="LLM Model Name")
        with gr.Row():
            openai_key = gr.Textbox(value=os.getenv("OPENAI_API_KEY", ""), label="OpenAI API Key", type="password")
            openai_baseurl = gr.Textbox(value=os.getenv("OPENAI_API_BASE", ""), label="OpenAI baseurl")
            ollama_host = gr.Textbox(value=os.getenv("OLLAMA_HOST", "http://localhost:11434"), label="Ollama Host")
            #ollama_host = gr.Textbox(value=os.getenv("OPENAI_API_EMBED_BASE", ""), label="Ollama Host")
            openai_baseurl_embed = gr.Textbox(value=os.getenv("OPENAI_API_EMBED_BASE", ""), label="OpenAI Embed baseurl")
            llm_model_embed = gr.Textbox(value=os.getenv("LLM_MODEL_EMBED",""), label="Embedding Model") #.split('/')[1], label="Embedding Model")
            openai_key_embed = gr.Textbox(value=os.getenv("OPENAI_API_KEY_EMBED", ""), label="OpenAI API Key Embed", type="password")  #("OLLAMA_API_KEY", ""), label="OpenAI API Key Embed", type="password")
        setup_btn = gr.Button("Initialise App")
        status_box = gr.Textbox(label="Status / Progress", interactive=True)  #interactive=False)
        with gr.Row():
            index_btn = gr.Button("Index Documents")
            stop_btn = gr.Button("Stop", variant="stop")  ## Add cancel event button
            query_text = gr.Textbox(label="Your Query")
            mode = gr.Dropdown(["naive", "local", "global", "hybrid", "mix"], value="hybrid", label="Query Mode")
            query_btn = gr.Button("Query")
        answer_box = gr.Markdown(label="Answer")
        kg_btn = gr.Button("Visualise Knowledge Graph")
        kg_html = gr.HTML(label="Knowledge Graph Visualisation")
        
        # Add progress tracking
        progress = gr.Textbox(label="Progress", interactive=False)
        
        # Button logic with async handling
        async def setup_wrapper(df, wd, llm, oai, base, base_embed, model, embed, host, embedkey):
            return await app_logic.setup(df, wd, llm, oai, base, base_embed, model, embed, host, embedkey)
            
        async def index_wrapper(df):
            return await app_logic.index_documents(df)
            
        async def query_wrapper(q, m):
            return await app_logic.query(q, m)
        
        def stop_wrapper():  ##SMY sync or async
            """Cancel event wrapper"""
            app_logic.trigger_cancel()
            return "Cancellation requested. Awaiting current step to finish..."
        
        # Button handlers
        ''' previous implementation before async coroutine err
        setup_btn.click(
            lambda df, wd, llm, oai, base, model, embed: app_logic.setup(df, wd, llm, oai, base, model, embed),
            [data_folder, working_dir, llm_backend, openai_key, openai_baseurl, llm_model_name, llm_model_embed],
            #[data_folder, llm_backend, openai_key, ollama_host, llm_model_name],
            status_box,
            )
        index_btn.click(
           lambda df: app_logic.index_documents(df),
                       [data_folder],
                       [status_box, progress], 
            )  
        query_btn.click(
            lambda q, m: app_logic.query(q, m),
                        [query_text, mode],
                        answer_box                        
            )
        kg_btn.click(
            lambda: app_logic.show_kg(),
            None,
            kg_html,
        )              
        '''
        '''
        ## setup() args:
        async def setup(self, data_folder: str, working_dir: str, llm_backend: str, 
             openai_key: str, openai_baseurl: str, openai_baseurl_embed: str, llm_model_name: str, 
             llm_model_embed: str, ollama_host: str, embed_key: str) -> str:
        '''
        setup_btn.click(
            fn=setup_wrapper,
            inputs=[data_folder, working_dir, llm_backend, openai_key, openai_baseurl, openai_baseurl_embed, llm_model_name, llm_model_embed, ollama_host, openai_key_embed],
            outputs=status_box,
            show_progress=True
        )
        index_btn.click(
            fn=index_wrapper,
            inputs=[data_folder],
            outputs=[status_box, progress],
            show_progress=True
        )
        query_btn.click(
            fn=query_wrapper,
            inputs=[query_text, mode],
            outputs=answer_box
        )
        kg_btn.click(
            fn=app_logic.show_kg,
            inputs=None,
            outputs=kg_html,
            show_progress=True
        )
        stop_btn.click(
            fn=stop_wrapper,
            inputs=[],
            outputs=[status_box]
        )
    return gradio_ui

if __name__ == "__main__":
    #gradio_ui().launch() 
    
    ##SMY: assist: https://www.gradio.app/guides/developing-faster-with-reload-mode
    ##SMY: NB: gradio app_gradio_lightrag.py --demo-name=gradio_ui
    async def main():
        try:
            app_logic = LightRAGApp()
            gradio_ui().launch()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if app_logic.rag:
                await app_logic.rag.finalize_storages()
    
    ##SMY Configure logging before running the main function: See lightrag_openai_compatible_demo.py
    configure_logging()
    
    asyncio.run(main())

    ##SMY: gradio reload-mode watch: https://github.com/huggingface/smolagents/issues/789
    #run_process(".", target=gradio_ui)