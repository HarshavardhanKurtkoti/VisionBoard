import os
import sys
from typing import Optional

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message including file name and line number
    Args:
        error: The exception that was raised
        error_detail: System module / sys providing exc_info
    Returns:
        str: Formatted error message
    """
    exc_info = error_detail.exc_info() if error_detail is not None else sys.exc_info()
    if exc_info is not None and len(exc_info) == 3 and exc_info[2] is not None:
        exc_tb = exc_info[2]
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in python script [{os.path.basename(file_name)}] line [{line_number}]: {str(error)}"
    
    return f"Error occurred: {str(error)}"

class VisionBoardException(Exception):
    """
    Custom exception class for VisionBoard project
    Attributes:
        error_message: Detailed error message
    """
    
    def __init__(self, error: Exception, error_detail: Optional[sys] = sys):
        """
        Initialize VisionBoardException with error details
        Args:
            error: The exception that was raised
            error_detail: System module providing exc_info
        """
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)
    
    def __str__(self) -> str:
        """
        String representation of the exception
        Returns:
            str: Error message
        """
        return self.error_message