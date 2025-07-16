'''
##############################################################
# Created Date: Wednesday, July 16th 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import functools
from tqdm import tqdm


def progress_bar_decorator(total_steps, description="Processing"):
    """
    A decorator to display a tqdm progress bar for a function.
    The decorated function is expected to accept the bar as its
    *second* positional argument.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # args[0] is self (for methods), so we inject pbar *after* that
            with tqdm(total=total_steps, desc=description) as pbar:
                return func(*args, pbar, **kwargs)
        return wrapper
    return decorator
