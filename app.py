import os   ## replace with Path
from pathlib import Path
import numpy as np  ##SMY

import gradio as gr
#from watchfiles import run_process  ##gradio reload watch
from app_gradio_lightrag import LightRAGApp  ##SMY lightrag logging
from utils.llm_login import get_login_token

import asyncio
import nest_asyncio
nest_asyncio.apply  #

import logging, logging.config

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

# Instantiate app logic
#app_logic = LightRAGApp()  ## See main()

# Gradio UI
def gradio_ui(app_logic: LightRAGApp):
    
    css_icon = """
    /* Make reveal button looks like an icon rather than a standard button */
    .password-box {
        position: relative;
        display: flex;
        align-items: center;
        min-width: 30;
    }
    .password-box > :first-child {
        flex-grow: 1;
    }
    .icon-button {
        position: absolute;
        right: 10px;
        top: 70%;
        transform: translateY(-50%);
        border: none;
        background: none;
        color: #4B4B4B;
        font-size: 1.2em;
        padding: 0;
        min-width: 0;
        box-shadow: none;
        cursor: pointer;
        z-index: 100;       /* on top */
    }
    """
              
    with gr.Blocks(theme=gr.themes.Soft(), title="SemmyKG - LightRAG Knowledge Graph App", css=css_icon) as gradio_ui: #demo:
        gr.Markdown("""
        # SemmyKG: LightRAG-based Knowledge Graph RAG
        Upload your markdown docs, index and build a knowledge graph, and query with OpenAI or Ollama. Visualise the KG interactively.
        """)

        # Step 0: Section 1
        # Define openai_api textbox initial value
        openai_api_key_init = os.getenv("OPENAI_API_KEY", "jan-ai")
        with gr.Accordion(label="🛞 LLM settings", open=False):
            with gr.Row():
                data_folder_tb = gr.Textbox(value="dataset/data/docs2", label="Data Folder (markdown only)", show_copy_button=True)
                working_dir_tb = gr.Textbox(value="./working_folder1", label="lightRAG working folder", show_copy_button=True)
                llm_backend_cb = gr.Radio(["OpenAI", "Ollama", "GenAI"], value="OpenAI", label="LLM Backend: OpenAI, Local or GenAI")
                llm_model_name_tb = gr.Textbox(value=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"), label="LLM Model Name", show_copy_button=True)  #.split('/')[1], label="LLM Model Name") "meta-llama/Llama-4-Maverick-17B-128E-Instruct")),  #image-Text-to-Text  #"openai/gpt-oss-120b",
            with gr.Row():
                with gr.Row():  #elem_classes="password-box"):
                    #openai_key_tb = gr.Textbox(value=os.getenv("OPENAI_API_KEY", "jan-ai"), label="OpenAI API Key", 
                    #                           type="password", elem_classes="password-box", container=False, interactive=True, info="OpenAI API Key") #, show_copy_button=True)
                    openai_key_tb = gr.Textbox(value=openai_api_key_init, label="OpenAI API Key", 
                                               type="password", elem_classes="password-box", container=False, interactive=True, info="OpenAI API Key") #, show_copy_button=True)
                    toggle_btn_openai_key = gr.Button(
                                value="👁️",  # Initial eye icon
                                elem_classes="icon-button", size="sm")  #, min_width=50)
                openai_baseurl_tb = gr.Textbox(value=os.getenv("OPENAI_API_BASE", "https://router.huggingface.co/v1"), label="OpenAI baseurl", show_copy_button=True)
                ollama_host_tb = gr.Textbox(value=os.getenv("OLLAMA_HOST", "http://localhost:1234/v1"), label="Ollama Host", show_copy_button=True)
                #ollama_host_tb = gr.Textbox(value=os.getenv("OPENAI_API_EMBED_BASE", ""), label="Ollama Host")
            with gr.Row():    
                embed_backend_dd = gr.Dropdown(choices=["Transformer", "Provider"], value="Provider", label="Embedding Type")
                openai_baseurl_embed_tb = gr.Textbox(value=os.getenv("OPENAI_API_EMBED_BASE", "http://localhost:1234/v1"), label="LLM Embed baseurl", show_copy_button=True)
                llm_model_embed_tb = gr.Textbox(value=os.getenv("LLM_MODEL_EMBED","text-embedding-bge-m3"), label="LLM Embedding Model", show_copy_button=True) #.split('/')[1], label="Embedding Model")
                with gr.Row():  #elem_classes="password-box"):
                    openai_key_embed_tb = gr.Textbox(value=os.getenv("OPENAI_API_KEY_EMBED", "jan-ai"), label="LLM API Key Embed",   #lm-studio
                                               type="password", elem_classes="password-box", container=False, interactive=True, info="LLM API Key Embed") #, show_copy_button=True)
                    toggle_btn_openai_key_embed = gr.Button(
                                value="👁️",  # Initial eye icon
                                elem_classes="icon-button", size="sm")  #, min_width=50)
                #openai_key_embed_tb = gr.Textbox(value=os.getenv("OPENAI_API_KEY_EMBED", "jan-ai"), label="OpenAI API Key Embed", type="password", show_copy_button=True)  #("OLLAMA_API_KEY", ""), label="OpenAI API Key Embed", type="password")
        
        # Step 1: Section 2
        with gr.Accordion("🤗 HuggingFace Client Control", open=True):  #, open=False):
            # HuggingFace controls
            hf_login_logout_btn = gr.LoginButton(value="Sign in to HuggingFace 🤗", logout_value="Logout of HF: ({}) 🤗", variant="huggingface")

        gr.Markdown("---")   #gr.HTML("<hr>")
        
        setup_btn = gr.Button("Initialise App", variant="primary")
        status_box = gr.Textbox(label="Status / Progress", interactive=True)  #interactive=False)
        
        # Step 2: Section 3
        gr.HTML("<hr>")   #gr.Markdown("---")
        
        with gr.Row():
            index_btn = gr.Button("Index Documents")
            stop_btn = gr.Button("Stop", variant="stop")  ## Add cancel event button
            query_text_tb = gr.Textbox(label="Your Query")
            mode_dd = gr.Dropdown(["naive", "local", "global", "hybrid", "mix"], value="hybrid", label="Query Mode")
            query_btn = gr.Button("Query")
        answer_box_md = gr.Markdown(label="Answer")
        
        # Step 3: Section 4
        kg_btn = gr.Button("Visualise Knowledge Graph")
        kg_html = gr.HTML(label="Knowledge Graph Visualisation")
        
        # Add progress tracking
        progress_tb = gr.Textbox(label="Progress", interactive=False)
        
        
        ##### Processing #####
        ## Note: 1.4.9 query `references` field, `user_prompt`  | lightRAG 1.4.0: QueryParam updated. Remove dependency on graspologic 

        # Initialise gr.State  ##gr.State component initial value must be able to be deepcopied
        st_openai_key = gr.State(value=openai_api_key_init)    #gr.State("")
        st_password1 = gr.State(value="password")
        st_password2 = gr.State(value="password")


        ### Change handling
        # Change Handling: update state value
        def update_state_stored_value(new_component_input):
            """ Updates stored state for Gradio Component
            for instance: st_openai_key.value = openai_key_tb
            Args:
                new_component_input: New value from component
            Returns:
                Updated value for state
            """ 
            return new_component_input
        
        # Change Handling: update Ollama
        def update_ollama(llm_backend):
            """ Update LLM settings fields with ollama values"""
            # Get model name excluding the model provider: # llm_model_name.rpartition("/")[-1]
            
            if llm_backend == "Ollama":
                return {
                #llm_backend_cb: gr.update(value="Ollama"),
                llm_model_name_tb: gr.update(value=os.getenv("LLM_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct").rpartition("/")[-1]),  #image-Text-to-Text  #"openai/gpt-oss-120b",  ##Text-to-Text)    #(value="llama2"),
                openai_key_tb: gr.update(value=os.getenv("OPENAI_API_KEY", "jan-ai"), info="LLM API Key"),
                openai_baseurl_tb: gr.update(value=os.getenv("OPENAI_API_BASE", "https://router.huggingface.co/v1")),
                ollama_host_tb: gr.update(value=os.getenv("OLLAMA_HOST", "http://localhost:1234/v1")), #"http://localhost:11434"
                openai_baseurl_embed_tb: gr.update(value=os.getenv("OPENAI_API_EMBED_BASE", "http://localhost:1234/v1")),   #"http://localhost:1234/v1/embeddings"
                llm_model_embed_tb: gr.update(value=os.getenv("LLM_MODEL_EMBED","nomic-embed-text")),
                openai_key_embed_tb: gr.update(value=os.getenv("OPENAI_API_KEY_EMBED", "jan-ai"))
                }
            elif llm_backend == "GenAI":
                return {
                llm_model_name_tb: gr.update(value=os.getenv("LLM_MODEL", "google/gemini-2.5-flash-preview-09-2025").rpartition("/")[-1]),  #image-Text-to-Text #"google/gemini-2.0-flash-exp:free" #"openai/gpt-oss-120b",  ##Text-to-Text)    #(value="llama2"),
                openai_key_tb: gr.update(value=os.getenv("GEMINI_API_KEY", "jan-ai"), info="GenAI API Key"),
                openai_baseurl_tb: gr.update(value=os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/"), label="GenAI baaseurl"),
                ollama_host_tb: gr.update(value=os.getenv("OLLAMA_HOST", "http://localhost:11434")), #"http://localhost:1234/v1"
                openai_baseurl_embed_tb: gr.update(value=os.getenv("OPENAI_API_EMBED_BASE", "http://localhost:1234/v1")),   #"http://localhost:1234/v1/embeddings"
                llm_model_embed_tb: gr.update(value=os.getenv("LLM_MODEL_EMBED","all-MiniLM-L6-v2")),
                openai_key_embed_tb: gr.update(value=os.getenv("OPENAI_API_KEY_EMBED", "jan-ai"))
                }
            elif llm_backend == "OpenAI":
                return {
                    llm_model_name_tb: gr.update(value=os.getenv("LLM_MODEL", "openai/gpt-oss-120b")),  #image-Text-to-Text  #"openai/gpt-oss-120b",  ##Text-to-Text)    #(value="llama2"),
                    openai_key_tb: gr.update(value=os.getenv("OPENAI_API_KEY", ""), info="OpenAI API Key"),
                    openai_baseurl_tb: gr.update(value=os.getenv("OPENAI_API_BASE", "https://router.huggingface.co/v1"), label="OpenAI baseurl"),
                    ollama_host_tb: gr.update(value=os.getenv("OLLAMA_HOST", "http://localhost:11434")), #"http://localhost:1234/v1"
                    openai_baseurl_embed_tb: gr.update(value=os.getenv("OPENAI_API_EMBED_BASE", "https://api.openai.com/v1")),   #"http://localhost:1234/v1/embeddings"
                    llm_model_embed_tb: gr.update(value=os.getenv("LLM_MODEL_EMBED","text-embedding-3-small")),
                    openai_key_embed_tb: gr.update(value=os.getenv("OPENAI_API_KEY_EMBED", ""))
                }
        
        # Change Handling: Update password reveal state - reusable function for toggling password visibility
        def toggle_password(current_state):
            """ Change state
            Change password input field between visible/hidden
            Args:
                current_state: Current password visibility state
            Returns:
                Tuple of updates for textbox, button and state
            """ 
            
            new_state = "text" if current_state == "password" else "password"
            new_icon = "👁️" if new_state == "password" else "👁️‍🗨️" # Change icon with state
            return [            #(
                gr.update(type=new_state),  #gr.Textbox.update(type=new_state),
                gr.update(value=new_icon),  #gr.Button.update(value=new_icon),
                new_state ]     #)
        
        # Update gr.State values on HF login change.
        def custom_do_logout(openai_key, oauth_token: gr.OAuthToken | None=None,):
            #'''  ##SMY: TO DELETE
            try:
                if oauth_token:
                    st_openai_key_get= update_state_stored_value(oauth_token.token)   ##SMY: currently not used optimally
            except AttributeError:
                st_openai_key_get= get_login_token(openai_key)  #(openai_key_tb)
            #'''            
            #return gr.update(value="Sign in to HuggingFace 🤗")
            return gr.update(value="Sign in to HuggingFace 🤗"), gr.update(value=st_openai_key_get)     #, gr.update(visible=True, value=msg)  #, state_api_token_arg

        
        # Button logic with async handling
        async def setup_wrapper(df, wd, llm_back, embed_back, oai, base, base_embed, model, model_embed, host, embedkey):
            return await app_logic.setup(df, wd, llm_back, embed_back, oai, 
                                         base, base_embed, model, model_embed, host, embedkey)
            
        async def index_wrapper(df):
            return await app_logic.index_documents(df)
            
        async def query_wrapper(q, m):
            return await app_logic.query(q, m)
        
        def stop_wrapper():  ##SMY sync or async
            """Cancel event wrapper"""
            app_logic.trigger_cancel()
            return "Cancellation requested. Awaiting current step to finish..."
        
        ### Change handlers
        llm_backend_cb.change(show_progress="hidden", fn=update_ollama, inputs=llm_backend_cb,  #inputs=None, 
                              outputs=[llm_model_name_tb, openai_key_tb, openai_baseurl_tb, ollama_host_tb, openai_baseurl_embed_tb, llm_model_embed_tb, openai_key_embed_tb])

        ### Button handlers

        #hf_login_logout_btn.click(update_state_stored_value, inputs=openai_key_tb, outputs=st_openai_key)
        hf_login_logout_btn.click(fn=custom_do_logout, inputs=openai_key_tb, outputs=[hf_login_logout_btn, st_openai_key])
        
        toggle_btn_openai_key.click(
            fn=toggle_password,
            inputs=[st_password1],
            outputs=[openai_key_tb, toggle_btn_openai_key, st_password1],
            show_progress="hidden"
            )
        toggle_btn_openai_key_embed.click(
            fn=toggle_password,
            inputs=[st_password2],
            outputs=[openai_key_embed_tb, toggle_btn_openai_key_embed, st_password2],
            show_progress="hidden"
            )
        
        inputs_arg = [data_folder_tb, working_dir_tb, llm_backend_cb, embed_backend_dd, st_openai_key, #openai_key_tb, 
                      openai_baseurl_tb, openai_baseurl_embed_tb, llm_model_name_tb, llm_model_embed_tb, 
                      ollama_host_tb, openai_key_embed_tb]
        
        setup_btn.click(
            fn=setup_wrapper,
            #inputs=[data_folder_tb, working_dir_tb, llm_backend_cb, openai_key_tb, openai_baseurl_tb, openai_baseurl_embed_tb, llm_model_name_tb, llm_model_embed_tb, ollama_host_tb, openai_key_embed_tb],
            inputs=inputs_arg,
            outputs=status_box,
            show_progress=True
            )
        index_btn.click(
            fn=index_wrapper,
            inputs=[data_folder_tb],
            outputs=[status_box, progress_tb],
            show_progress=True
            )
        query_btn.click(
            fn=query_wrapper,
            inputs=[query_text_tb, mode_dd],
            outputs=answer_box_md
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
        from app_gradio_lightrag import LightRAGApp
        # Instantiate LightRAG and launch Gradio
        try:
            app_logic = LightRAGApp()
            gradio_ui(app_logic).launch(server_port=7866)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if app_logic.rag:
                await app_logic.rag.finalize_storages()
 
    from utils.logger import get_logger, setup_logging
    setup_logging()         ## set logging
    logger_kg = get_logger("semmyKG")   ## app logging

    ##SMY Initialise logging before running the main function: See lightrag_openai_compatible_demo.py
    from app_gradio_lightrag import handle_errors, configure_logging
    configure_logging()     ## lightRAG logging
    
    asyncio.run(main())

    ##SMY: gradio reload-mode watch: https://github.com/huggingface/smolagents/issues/789
    #run_process(".", target=gradio_ui)