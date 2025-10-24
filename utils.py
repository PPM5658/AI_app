# utils.py
# Helper functions

def format_time(seconds):
    """Converts seconds to a 0.00m or 0.00s format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
