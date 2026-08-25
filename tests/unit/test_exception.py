import sys
import pytest
from visionboard.exception.exception import VisionBoardException, error_message_detail

def test_visionboard_exception_in_active_traceback():
    """Test VisionBoardException formatting when raised inside try-except"""
    try:
        1 / 0
    except Exception as e:
        vb_err = VisionBoardException(e, sys)
        msg = str(vb_err)
        assert "Error occurred in python script" in msg
        assert "division by zero" in msg
        assert "line [" in msg

def test_visionboard_exception_without_active_traceback():
    """Test VisionBoardException when sys.exc_info() has no traceback"""
    err = ValueError("Manual error message")
    vb_err = VisionBoardException(err, None)
    msg = str(vb_err)
    assert "Manual error message" in msg
