##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################


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
