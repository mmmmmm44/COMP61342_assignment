from datetime import datetime

def print_log(message: str) -> None:
    """
    Print a log message with a timestamp.

    Args:
        message (str): The message to log.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')[:-3]}] - {message}")