'''
class to host Control element of Abstract scenario
'''


class Control:
    """Control class to host the signal element of the abstract scenario.
    """
    def __init__(self):
        """Initialize the Control class with the signal element as an empty dictionary."""
        self.Signal = {}

    def isEmpty(self):
        """Check if the Control element is empty."""
        return not bool(self.Signal)
