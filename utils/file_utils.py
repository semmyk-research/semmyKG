# file_handler/file_utils.py

from pathlib import Path
import sys
import shutil
import tempfile

from itertools import chain
from typing import List, Optional, Union, Any, Mapping
from PIL import Image


def check_create_logfile(filename: str, dir_path: Union[str, Path]="logs", tz_hours=None, date_format="%Y-%m-%d") -> Path:
    """
    check if log file exists, else create one and return the file path.

    Args:
        directory_path (str): The path to the directory.
        filename (str): The name of the file to check/create.
    Returns:
        The pathlib.Path object for the file
    """

    import datetime
    import warnings
    import tempfile
    from utils.utils import get_time_now_str

    # 1. Get the path of the current script's parent directory (the project folder).
    # `__file__` is a special variable that holds the path to the current script.
    #project_root = Path(__file__).resolve().parent.parent
    project_root = Path(__file__).resolve().parents[1]   ##SMY: parents[1] gets the second-level parent (the grandparent)
    
    # 2. Define the path for the logs directory.
    # The `/` operator is overloaded to join paths easily.
    writable_dir = project_root / dir_path if isinstance(dir_path, str) else Path(dir_path)
    
    # 3. Define and create the logs directory if it doesn't already exist.
    # `mkdir()` with `exist_ok=True` prevents a FileExistsError if the folder exists.

    logs_dir = writable_dir     #project_root / dir_path
     
    try:
        if not logs_dir.is_dir():
            logs_dir.mkdir(mode=0o2755, parents=True, exist_ok=True)
    except PermissionError: ##[Errno 13] Permission denied: '/home/user/app/logs/app_logging_2025-09-18.log'
        warnings.warn("[Errno 13] Permission denied, possibly insufficient permission or Persistent Storage not enable: attempting chmod 0o2644")
        #writable_dir = Path(tempfile.gettempdir())    # 
        writable_dir.mkdir(mode=0o2755, parents=True, exist_ok=True)
        writable_dir.chmod(0o2755)
        if not writable_dir.is_dir():
            warnings.warn(f"Working without log files in directory: {writable_dir}")
    
    # 4. Create log file with a timestamp inside the new logs directory.
    # This ensures a unique log file is created for the day the script runs.
    #timestamp = datetime.datetime.now().strftime("%Y-%m-%d")  #.strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = get_time_now_str(tz_hours=tz_hours, date_format="%Y-%m-%d")
    log_file = logs_dir / f"{Path(filename).stem}_{timestamp}.log"
    
    # 5. Check if the file exists (it won't, if it's not the same day).
    if not log_file.exists(): # or log_file.is_file():
        # If the file doesn't exist, touch() will create an empty file.
        log_file.touch(exist_ok=True)
    
    #print(f"Created log file at: {log_file}")  ##debug

    return log_file 

## debug
'''
from pathlib import Path
from typing import Union
resolve_grandparent_object("file_handler")
print(f'file: {check_create_logfile("app_logging.log")}')
'''

##SMY: - Make generic for any file apart from log files
def check_create_dir(dir_name: Union[str, Path]) -> Path:
    """
    check if directory exists, else create one and return the directory path.

    Args:
        directory_path (str): The path to the directory.
        filename (str): The name of the directory to check/create.
    Returns:
        The pathlib.Path object for the directory
    """

    import warnings
    
    try:
        dir_path = Path(dir_name)
        #if dir_path.is_dir():
        #    dir_path.mkdir(parents=True, exist_ok=True)  #, mode=0o2755)
        dir_path.mkdir(parents=True, exist_ok=True)  #, mode=0o2755)
    except PermissionError: ##[Errno 13] Permission denied: '/home/user/app/logs/app_logging_2025-09-18.log'
        warnings.warn("[Errno 13] Permission denied, possibly insufficient permission or Persistent Storage not enable: attempting chmod 0o2644")
        dir_path.mkdir(mode=0o2755, parents=True, exist_ok=True)
        dir_path.chmod(0o2755)
    
    return dir_path

def check_create_file(filename: Union[str, Path]) -> Path:
    """
    check if File exists, else create one and return the file path.

    Args:
        directory_path (str): The path to the directory.
        filename (str): The name of the file to check/create.
    Returns:
        The pathlib.Path object for the file
    """

    import warnings
    
    try:
        filename_path = Path(filename)
        filename_path.touch(exist_ok=True)  #, mode=0o2755)
    except PermissionError: ##[Errno 13] Permission denied: '/home/user/app/logs/app_logging_2025-09-18.log'
        warnings.warn("[Errno 13] Permission denied, possibly insufficient permission or Persistent Storage not enable: attempting chmod 0o2644")
        filename_path.touch(exist_ok=True, mode=0o2755)  # Creates an empty file if it doesn't exists
        filename_path.chmod(0o2755)
    
    return filename_path

def create_temp_folder(tempfolder: Optional[str | Path] = '', program_name: str = "semmyKG"):
    """ Create a temp folder Gradio and output_dir if supplied"""

    # Create a temporary directory in a location where Gradio can access it.
    output_dir = check_create_dir(Path(tempfile.gettempdir()) / f"{program_name}_temp_output"/ tempfolder if tempfolder else Path(tempfile.gettempdir()) / f"{program_name}_temp_output")
    
    return output_dir

def accumulate_dir(uploaded_files, current_state, ext: Union[str, tuple] = (".md", "md")):
    """ accumulate uploaded files in dir based on ext with the existing state """

    import gradio as gr

    # Initialise state if it's the first run
    if current_state is None:
        current_state = []

    # Check if files were uploaded in the current iteration, return the current state.
    if not uploaded_files:
        return current_state, gr.update(), gr.update(visible=True, value="No new files uploaded"), gr.update(value="No new files uploaded")

    # call is_file_with_extension to check if pathlib.Path object is a file and has a non-empty extension
    #new_file_paths = [f.name for f in uploaded_files if is_file_with_extension(Path(f.name))]  #Path(f.name) and Path(f.name).is_file() and bool(Path(f.name).suffix)]  #Path(f.name).suffix.lower() !=""]
    new_file_paths = [f.name for f in uploaded_files if is_file_with_extension(Path(f.name)) and f.name.endswith(ext)] 
    
    # Concatenate the new files with the existing ones in the state
    updated_files = current_state + new_file_paths
    updated_filenames = [Path(f).name for f in updated_files]      ##SMY: filenames only

    updated_files_count = len(updated_files)
    
    # Return the updated state and a message to the user
    filename_info = "\n".join(updated_filenames)    ##SMY: not used(updated_filenames)
    #message = f"Accumulated {len(updated_files)} file(s) total: \n{filename_info}"
    message_count = f"Accumulated {updated_files_count} file(s) total."
    message = f"Accumulated {updated_files_count} file(s) total: \n{filename_info}"
    
    
    #outputs=[state_uploaded_file_list, dir_btn, upload_count_md, status_box],
    #return updated_files, updated_files_count, message, gr.update(interactive=True), gr.update(interactive=True)
    return updated_files, gr.update(interactive=True,), gr.update(visible=True, value=message_count), gr.update(value=message)


##========= 
def find_file(file_name: str) -> Path:  #configparser.ConfigParser:
    """
    Finds file from the same directory, parent's sibling or grandparent directory of the calling script.
    
    Args:
        file_name: The name of the file to find.
    
    Returns:
        The path of the file.
    
    Raises:
        FileNotFoundError: If the file cannot be found.
    Drawback:
        Return the first result from the for loop iteration through the generator produced by rglob(). As soon as the first match is found, the function returns it, making the process very efficient by not searching any further but might not match the exact. 
    """
    
    # 1. Get the current script's path, its parent and its grandparent directory
    # Start the search from the directory of the file this function is in
    try:
        current_path = Path(sys.argv[0]).resolve()
    except IndexError:        
        # Handle cases where sys.argv[0] might not exist (e.g., in some IDEs)
        current_path = Path(__file__).resolve()
        #current_path = Path('.').resolve()  ##unreliable

    parent_dir = current_path.parent
    grandparent_dir = current_path.parent.parent

    # Walk up the directory tree until the config file is found
    '''
    for parent in [current_path, *current_path.parents]:
        config_path = parent / file_name
        if config_path.is_file():
            return config_path
    raise FileNotFoundError(f"Configuration file '{file_name}' not found.")
    '''
    try:
        # 1. Search the parent directory directly
        parent_filepath = parent_dir / file_name
        if parent_filepath.is_file():
            return parent_filepath
            
        # 2. Search the grandparent directory directly
        grandparent_filepath = grandparent_dir / file_name
        if grandparent_filepath.is_file():
            return grandparent_filepath

        # 3. Search recursively in all subdirectories of the grandparent.
        #    This will cover all sibling directories of the parent.
        for p in grandparent_dir.rglob(file_name):
            if p.is_file():
                return p
        
        return None
    except Exception as exc:
        return exc

def resolve_grandparent_object(gp_object:str):
    ###
    # Create a Path object based on current file's location, resolve it to an absolute path,
    # and then get its parent's parent using chained .parent calls or the parents[] attribute.

    #import sys
    
    # 1. Get the current script's path, its parent and its grandparent directory
    try:
        current_path = Path(sys.argv[0]).resolve()        
    except IndexError:        
        # Handle cases where sys.argv[0] might not exist (e.g., in some IDEs)
        current_path = Path(__file__).resolve()
        #current_path = Path('.').resolve()    ##unreliable

    parent_dir = current_path.parent
    grandparent_dir = current_path.parent.parent
    
    #grandparent_dir = Path(__file__).resolve().parent.parent

    sys.path.insert(0, f"{grandparent_dir}")  #\\file_handler")
    sys.path.insert(1, f"{grandparent_dir}\\{gp_object}")
    #print(f"resolve: sys.path[0]:  {sys.path[0]}")  ##debug
    #print(f"resolve: sys.path[1]:  {sys.path[1]}")  ##debug



def zip_processed_files(root_dir: str, file_paths: list[str], tz_hours=None, date_format='%d%b%Y_%H-%M-%S') -> Path:
    """
    Creates a zip file from a list of file paths (strings) and returns the Path object.
    It preserves the directory structure relative to the specified root directory.

    Args:
        root_dir (str): The root directory against which relative paths are calculated.
        file_paths (list[str]): A list of string paths to the files to be zipped.

    Returns:
        str(Path): The string of the Path object of the newly created zip file.
    """

    import zipfile
    from utils import file_utils
    from utils import utils

    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise ValueError(f"Root directory does not exist: {root_path}")

    # Create a temporary directory in a location where Gradio can access it.
    ##SMY: synced with create_temp_folder()
    '''gradio_output_dir = Path(tempfile.gettempdir()) / "gradio_temp_output"
    #gradio_output_dir.mkdir(exist_ok=True)
    file_utils.check_create_dir(gradio_output_dir)
    final_zip_path = gradio_output_dir / f"outputs_processed_{utils.get_time_now_str(tz_hours=tz_hours, date_format=date_format)}.zip"
    '''
    final_zip_path = Path(root_dir).parent / f"outputs_processed_{utils.get_time_now_str(tz_hours=tz_hours, date_format=date_format)}.zip"
    
    # Use a context manager to create the zip file: use zipfile() opposed to shutil.make_archive
    # 'w' mode creates a new file, overwriting if it already exists. 
    zip_unprocessed = 0
    with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists() and file_path.is_file():
                # Calculate the relative path from the root_dir.
                # The `arcname` parameter tells `zipfile` what the path inside the zip file should be.
                arcname = file_path.relative_to(root_path)
                zipf.write(file_path, arcname=arcname)
            else:
                #print(f"Warning: Skipping {file_path_str}, as it is not a valid file.")
                zip_processed_files += 1  ##SMY:future - to be implemented

    #return final_zip_path
    return str(final_zip_path)


def process_and_zip(input_dir_path):
    """
    Finds dynamic directories, copies files from a source directory to a temporary directory, zips it,
    and returns the path to the zip file.
    
    Args:
        input_dir_path (str): The path to the directory containing files to be processed.
    
    Returns:
        pathlib.Path: The path to the generated zip file.
    """
    # Convert the input path to a Path object
    #input_path = Path(input_dir_path)
    parent_input_path = Path(input_dir_path)  #.parent

    # Check if the input directory exists
    if not parent_input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {parent_input_path}")
        
    # Create a temporary directory using a context manager
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        
        # Define the path for the output structure inside the temporary directory
        temp_output_path = temp_dir_path / "output_dir"
        
        # Copy all extracted files to the temporary directory
        # We use semantic accurate and performant .iterdir than more robust glob to get all files and folders
        
        for input_subdir in parent_input_path.iterdir():
            if input_subdir.is_dir():
            # Create the corresponding subdirectory in the temp directory
                temp_output_subdir = temp_output_path / input_subdir.name
                #temp_output_subdir.mkdir(parents=True, exist_ok=True)   #, mode=0o2755)
                #file_handler.file_utils.check_create_dir(temp_output_subdir)
                check_create_dir(temp_output_subdir)

                # Copy the files from the source subdirectory to the temp subdirectory
                #for item_path in input_path.glob('*'):
                for item_path in input_subdir.iterdir():    
                    if item_path.is_dir():
                        shutil.copytree(src=item_path, dst=temp_output_subdir / item_path.name)
                    else:
                        shutil.copy2(item_path, temp_output_subdir)
        
        # Create the zip file from the temporary directory
        zip_base_name = temp_dir_path / "outputs_processed_files"
        zip_file_path = shutil.make_archive(
            base_name=str(zip_base_name),   ##zip file's name
            format='zip',
            root_dir=str( temp_output_path)  #(temp_dir_path)     ##exclude from the archive
        )
        # Manually move the completed zip file to the Gradio-managed temporary directory
        final_zip_file_path = parent_input_path / Path(zip_file_path).name
        shutil.move(src=zip_file_path, dst=final_zip_file_path)
        
    # The shutil function returns a string, so we convert it back to a Path object in gr.File
    return str(final_zip_file_path)


def is_file_with_extension(path_obj: Path) -> bool:
    """
    Checks if a pathlib.Path object is a file and has a non-empty extension.
    """
    path_obj = path_obj if isinstance(path_obj, Path) else Path(path_obj) if isinstance(path_obj, str) else None
    return path_obj.is_file() and bool(path_obj.suffix)


def accumulate_files(uploaded_files, current_state):
    """
    Accumulates newly uploaded files with the existing state.
    """

    from globals_config import config_load
    import gradio as gr 
    # Initialise state if it's the first run
    if current_state is None:
        current_state = []
    
    # If no files were uploaded in this interaction, return the current state unchanged
    if not uploaded_files:
        return current_state, f"No new files uploaded. Still tracking {len(current_state)} file(s)."
    
    # Get the temporary paths of the newly uploaded files
    # call is_file_with_extension to check if pathlib.Path object is a file and has a non-empty extension
    #new_file_paths = [f.name for f in uploaded_files if is_file_with_extension(Path(f.name))]  #Path(f.name) and Path(f.name).is_file() and bool(Path(f.name).suffix)]  #Path(f.name).suffix.lower() !=""]
    new_file_paths = [f.name for f in uploaded_files if is_file_with_extension(Path(f.name)) and f.name.endswith(config_load.file_types_tuple)] 
    
    # Concatenate the new files with the existing ones in the state
    updated_files = current_state + new_file_paths
    updated_filenames = [Path(f).name for f in updated_files]

    updated_files_count = len(updated_files)
    
    # Return the updated state and a message to the user
    #file_info = "\n".join(updated_files)
    filename_info = "\n".join(updated_filenames)
    #message = f"Accumulated {len(updated_files)} file(s) total.\n\nAll file paths:\n{file_info}"
    message = f"Accumulated {len(updated_files)} file(s) total: \n{filename_info}"
    
    return updated_files, updated_files_count, message, gr.update(interactive=True), gr.update(interactive=True)

##NB: Python =>3.10, X | Y equiv to the type checker as Union[X, Y]
def collect_pdf_html_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    patterns = ["*.pdf", "*.html"]  #, "*.htm*"]
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    #pdfs_htmls = [p for p in root.rglob("*.pdf", "*.html", "*.htm*") if p.is_file()]
    #pdfs_htmls = [chain.from_iterable(root.rglob(pattern) for pattern in patterns)]
    # Use itertools.chain to combine the generators from multiple rglob calls
    pdfs_htmls = list(chain.from_iterable(root.rglob(pattern) for pattern in patterns))

    return pdfs_htmls

def collect_pdf_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    pdfs = [p for p in root.rglob("*.pdf") if p.is_file()]
    return pdfs

def collect_html_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all PDF files.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path {root} does not exist.")
    htmls = [p for p in root.rglob("*.html", ".htm") if p.is_file()]

    ## SMY: TODO: convert htmls to PDF. Marker will by default attempt weasyprint which typically raise 'libgobject-2' error on Win
    
    return htmls

def collect_markdown_paths(root: Union[str, Path]) -> List[Path]:
    """
    Recursively walk *root* and return a list of all Markdown files.
    """
    root = Path(root)
    md_files = [p for p in root.rglob("*.md") if p.is_file()]
    return md_files

def process_dicts_data(data:Union[dict, list[dict]]):
    """ Returns formatted JSON string for a single dictionary or a list of dictionaries"""
    import json
    from pathlib import Path   #WindowsPath
    #from typing import dict, list

    # Serialise WindowsPath objects to strings using custom json.JSoNEncoder subclass
    class PathEncoder(json.JSONEncoder):
        def default(self, obj):
            #if isinstance(obj, WindowsPath):
            if isinstance(obj, Path):    
                return str(obj)
            # Let the base class default method raise the TypeError for other types
            #return json.JSONEncoder.default(self, obj)
            return super().default(obj) # Use super().default() for better inheritance

    # Convert the list of dicts to a formatted JSON string
    formatted_string = json.dumps(data, indent=4, cls=PathEncoder)

    '''
    def path_to_str(obj):
        """
        A simple function to convert pathlib.Path objects to strings.
        """
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Convert the list of dicts to a formatted JSON string
    formatted_string = json.dumps(data, indent=4, default=path_to_str)
    '''
    
    return formatted_string