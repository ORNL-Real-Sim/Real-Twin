'''
class to host supply element of Concrete scenario
'''


class Supply:
    '''The supply class to host the supply element of Concrete scenario
    '''
    def __init__(self):
        self.NetworkName = {}
        self.Network = {}
        self.NetworkWithElevation = {}

    def is_empty(self):
        """Check if the Supply object is empty."""
        pass

    def generate_network(self, AbsScn):
        """Generate network data from the abstract scenario."""
        self.NetworkName = AbsScn.Network.NetworkName
        self.Network = AbsScn.Network.OpenDriveNetwork.OpenDrive_network[0]
        self.NetworkWithElevation = AbsScn.Network.OpenDriveNetwork.OpenDrive_network[1]
