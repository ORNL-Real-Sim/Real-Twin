'''
class to host demand element of Concrete scenario
'''


class TrafficControl:
    '''The demand class to host the demand element of Concrete scenario
    '''
    def __init__(self):
        self.Signal = {}

    def is_empty(self):
        """Check if the TrafficControl object is empty."""
        pass

    def generate_control(self, AbsScn):
        """Generate control data from the abstract scenario."""
        # =================================
        # load class from AbstractScenario
        # =================================
        # SignalDict = AbsScn.dataObjDict['Control']['Signal'].Signal
        # # Read this for now
        # IDRef = pd.read_csv('Synchro_lookuptable.csv')
        # MergedDf3 = pd.merge(SignalDict ,IDRef, on=['INTID'], how='left')

        pass
