# opted for sigleton as opposed to global variable

# Create a singleton object to hold all shared states
# This ensures that only one instance of the Config class is ever created
class Config:
    """ Single model_dict use across the app"""
    def __init__(self):
        self.model_dict = {}
        self.config_ini = "utils\\config.ini"
        self.output_dir = ""
        
        # File types
        self.file_types_list  = []
        self.file_types_tuple = (".pdf", ".html", ".docx", ".doc")

        # all other variables shared across the app 
        #self.pdf_files: list[str] = []
        #self.pdf_files_count: int = 0
        self.provider: str = ""
        self.model_id: str = ""
        #base_url: str
        self.hf_provider: str = ""
        self.endpoint: str = ""
        self.backend_choice: str = ""
        self.system_message: str = ""
        self.max_tokens: int = 8192
        self.temperature: float = 1.0
        self.top_p: float = 1.0
        self.stream: bool = False
        self.api_token: str = ""
        self.openai_base_url: str = "https://router.huggingface.co/v1"
        self.openai_image_format: str = "webp"

        self.tz_hours: float = 0.0  #:str = None
        #oauth_token: gr.OAuthToken | None=None,
        #progress: gr.Progress = gr.Progress(track_tqdm=True),  #Progress tracker to keep tab on pool queue executor


# Create a single, shared instance of the Config class
# Other modules will import and use this instance.
config_load_models = Config()
config_load = Config()

#if __name__ == "__main__":
    
