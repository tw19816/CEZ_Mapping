import os
class Config:
    """Database for configuring pipline
    
    Parameters:
        root_path (str) : Absolute path to project root dir.
    """
    root_path = os.path.split(os.path.split(__file__)[0])[0]