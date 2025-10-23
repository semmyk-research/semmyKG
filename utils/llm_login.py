from huggingface_hub import login, logout, get_token, whoami, HfApi
import os
#import traceback
from time import sleep
from typing import Optional

from utils.logger import get_logger

## Get logger instance
logger = get_logger(__name__)

def disable_immplicit_token():
    # Disable implicit token propagation for determinism
    # Explicitly disable implicit token propagation; we rely on explicit auth or env var
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

#def get_login_token( api_token_arg, oauth_token: gr.OAuthToken | None=None,):
def get_login_token( api_token_arg, oauth_token):
    """ Use user's supplied token or Get token from logged-in users, else from token stored on the  machine. Return token"""
    #oauth_token = get_token() if oauth_token is not None else api_token_arg
    if api_token_arg != '':  # or not None:  #| None:
        oauth_token = api_token_arg
    elif oauth_token:
        oauth_token = oauth_token.token
    else: oauth_token = '' if not get_token() else get_token()
    
    #return str(oauth_token) if oauth_token else ''  ##token value or empty string
    return oauth_token if oauth_token else ''  ##token value or empty string

def login_huggingface(token: Optional[str] = None):
    """
    Login to Hugging Face account. Prioritise CLI login for privacy and determinism.

    Attempts to log in to Hugging Face Hub.
    First, it tries to log in interactively via the Hugging Face CLI.
    If that fails, it falls back to using a token provided as an argument or
    found in the environment variables HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.

    If both methods fail, it logs a warning and continues without logging in.
    """

    logger.info("Attempting Hugging Face login...")
        
    # Disable implicit token propagation for determinism
    disable_immplicit_token()

    token = token
    # Privacy-first login: try interactive CLI first; fallback to provided/env token only if needed
    try:
        if whoami():  ##SMY: Call HF API to know "whoami".
            logger.info("✔️ hf_login already: whoami()", extra={"mode": "HF Oauth"})
            #return True
        else:
            login()   ##SMY: Not visible/interactive to users on HF Space. ## ProcessPoll limitation
            sleep(5)  ##SMY pause for login. Helpful: pool async opex 
            logger.info("✔️ hf_login already: login()", extra={"mode": "cli"})
            #return True
    except Exception as exc:
        # Respect common env var names; prefer explicit token arg when provided
        fallback_token = token if token else get_token() or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")  ##SMY: to revisit
        if fallback_token:
            try:
                login(token=fallback_token)
                #token = fallback_token  ##debug
                logger.info("✔️ hf_login through fallback", extra={"mode": "token"})  ##SMY: This only displays if token is provided
            except Exception as exc_token:
                logger.warning("❌ hf_login_failed through fallback", extra={"error": str(exc_token)})
        else:
            logger.warning("❌ hf_login_failed", extra={"error": str(exc)})
            # Silent fallback; client will still work if token is passed directly