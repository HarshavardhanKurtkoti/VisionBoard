import os
import sys
from visionboard.logging import logger

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate detailed error message including file name and line number
    Args:
        error: The exception that was raised
        error_detail: System information about the error
    Returns:
        str: Formatted error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script name [{os.path.basename(file_name)}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class VisionBoardException(Exception):
    """
    Custom exception class for VisionBoard project
    Attributes:
        error_message: Detailed error message
        error_detail: System information about the error
    """
    
    def __init__(self, error: Exception, error_detail: sys):
        """
        Initialize VisionBoardException with error details
        Args:
            error: The exception that was raised
            error_detail: System information about the error
        """
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)
    
    def __str__(self) -> str:
        """
        String representation of the exception
        Returns:
            str: Error message
        """
        return self.error_message

if __name__ == '__main__':
    try:
        logger.logging.info("Enter the try block")
        a = 1/0
        print("This will not be printed", a)
    except Exception as e:
        raise VisionBoardException(e, sys)