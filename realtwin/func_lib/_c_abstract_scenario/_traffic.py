'''
class to host Traffic element of Abstract scenario
'''


class Traffic:
    '''The traffic class to host the Traffic element of Abstract scenario'''
    def __init__(self):
        self.AllTrafficTypes = ['Volume', 'TurningRatio', 'Trajectory']
        self.Volume = None
        self.TurningRatio = None
        self.Trajectory = {}
        self.VolumeLookupTable = None

    def isEmpty(self):
        """Check if the Traffic element is empty"""
        pass
