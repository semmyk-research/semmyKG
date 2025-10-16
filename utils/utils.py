
def is_dict(variable):
    """Checks if a variable is a dict."""
    if isinstance(variable, dict):
        return True
        
    return False

def is_list_of_dicts(variable):
    """Checks if a variable is a list containing only dicts."""
    
    if isinstance(variable, list):
        # Return True only if the list is empty or all elements are dicts.
        return all(isinstance(item, dict) for item in variable)
        
    return False

def is_int(variable):
    """Checks if a variable is an interger."""
    if isinstance(variable, int):
        return True
        
    return False

def is_float(variable):
    """Checks if a variable is a float."""
    if isinstance(variable, float):
        return True
        
    return False

def get_time_now_str(tz_hours: float=0.0, date_format: str='%Y-%m-%d'):  #date_format:str='%d%b%Y'):
    """Returns the current time (offset from UTC) in a specific format + local time: ("%Y-%m-%d %H:%M:%S.%f %Z")."""
    from datetime import datetime, timezone, timedelta

    try:
        # Get the current time or UTC time
        if tz_hours or is_int(tz_hours):  #or tz_hours == 0.0 or tz_hours == float(0)
            current_utc_time = datetime.now(tz=timezone.utc)
            current_time = current_utc_time + timedelta(hours=float(tz_hours))
        else:
            current_time = datetime.now(tz=timezone.utc)
    except ValueError:
        current_utc_time = datetime.now(tz=timezone.utc)
        current_time = current_utc_time + timedelta(hours=float(0))

    # Format the time as a string
    #formatted_time = current_utc_time.strftime(date_format)  #("%Y-%m-%d %H:%M:%S.%f %Z")
    formatted_time = current_time.strftime(date_format)  #("%Y-%m-%d %H:%M:%S.%f %Z")

    #print(f"Current time: {formatted_time}")   ##debug
    return formatted_time

#get_time_now_str() ##debug
#print(f"time: {get_time_now_str(0)}")  ##debug