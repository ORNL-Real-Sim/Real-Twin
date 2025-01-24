'''
class to host a unique AbstractScenario definition
'''

# import four elements of AbstractScenario
from ._supply import Supply
from ._demand import Demand
from ._behavior import Behavior
from ._route import Route
from ._trafficControl import TrafficControl


class ConcreteScenario:
    """Initialize and Generate Concrete Scenario from Abstract Scenario"""
    def __init__(self):
        self.Supply = Supply()
        self.Demand = Demand()
        self.Behavior = Behavior()
        self.Route = Route()
        self.TrafficControl = TrafficControl()

    def is_empty(self):
        """Check if the ConcreteScenario object is empty."""
        pass

    def get_unified_scenario(self, AbsScn):
        """Generate Concrete Scenario from Abstract Scenario"""
        # copy the config_dict from AbstractScenario incase it is needed
        self.config_dict = AbsScn.config_dict

        # generate concrete scenario
        self.Supply.generate_network(AbsScn)
        self.Demand.generate_traffic(AbsScn)
        self.Behavior.ApplicationInterpreter(AbsScn)
        self.Route.generate_route(AbsScn)
        self.TrafficControl.generate_control(AbsScn)
