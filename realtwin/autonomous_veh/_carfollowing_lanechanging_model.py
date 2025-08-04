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

"""Car-Following Models and Lane-Changing Models for Autonomous Vehicles in SUMO."""

CFmodel = {
    'Human': {},
    'AVnormal': {},
    'AVsafe': {},
    'AVaggressive': {},
}

CF_DEFAULT_PARAMETERS = {
    "Wied99": {'cc0': 4.92,
               'cc1': 2,
               'cc2': 13.12,
               'cc3': -8,
               'cc4': -0.35,
               'cc5': 0.35,
               'cc6': 11.44,
               'cc7': 0.82,
               'cc8': 11.48,
               'cc9': 4.92},
    "Krauss": {'minGap': 0.5,
               'accel': 3.8,
               'decel': 4.5,
               'sigma': 0,
               'tau': 0.4,
               'emergencyDecel': 9},
    "IDM": {'minGap': 2.5,
            'accel': 2.6,
            'decel': 4.5,
            'tau': 1,
            'emergencyDecel': 9,
            'delta': 4},
    "CoopVisWied99": {'minLookAheadDist': 0,
                      'maxLookAheadDist': 820.21,
                      'minLookBackDist': 0,
                      'maxLookBackDist': 492.13,
                      'noOfInteractObjects': 2},
    "LCVisWied99": {'safetyDistRedFact': 0.45,
                    'coopLaneChange': True,
                    'minClearance': 1.64},
    "LCSumoKrauss": {'LcStrategic': 1,
                     'LcCooperative': 1,
                     'LcAccelLat': 1},
    "LCSumoIDM": {'LcStrategic': 1,
                  'LcCooperative': 1}
}

# CC0: Standstill distance. The desired gap between two vehicles in a stopped condition.
# CC1: Time headway/gap. The time headway(gap) a following driver maintains for safety when moving.
# CC2: Car-following distance/following variation. The variation in following distance.
# CC3: Threshold for entering following. The point at which a driver begins decelerating after perceiving a slower-moving leader and initiates an unconscious following behavior.
# CC4: Negative following threshold. Controls speed differences when opening the gap(negative relative speed).
# CC5: Positive following threshold. Controls speed differences when closing the gap(positive relative speed).
# CC6: Speed dependency of oscillation. The influence of distance on speed oscillation during following.
# CC7: Oscillation acceleration. Actual acceleration during oscillation in unconscious following.
# CC8: Standstill acceleration. Desired acceleration when starting from a standstill.
# CC9: Acceleration at 50mph. Desired acceleration at higher speeds, though limited by vehicle type's maximum acceleration

# Wied 99
CFmodel['Human']['Wied99'] = {'cc0': 4.92,
                              'cc1': 2,
                              'cc2': 13.12,
                              'cc3': -8,
                              'cc4': -0.35,
                              'cc5': 0.35,
                              'cc6': 11.44,
                              'cc7': 0.82,
                              'cc8': 11.48,
                              'cc9': 4.92}

CFmodel['AVnormal']['Wied99'] = {'cc0': 3.28,
                                 'cc1': 1,
                                 'cc2': 5.44,
                                 'cc3': -8,
                                 'cc4': -0.35,
                                 'cc5': 0.35,
                                 'cc6': 0,
                                 'cc7': 1.08,
                                 'cc8': 12.46,
                                 'cc9': 5.9}

CFmodel['AVsafe']['Wied99'] = {'cc0': 3.28,
                               'cc1': 1,
                               'cc2': 5.44,
                               'cc3': -8,
                               'cc4': -0.35,
                               'cc5': 0.35,
                               'cc6': 0,
                               'cc7': 1.08,
                               'cc8': 12.46,
                               'cc9': 5.9}

CFmodel['AVaggressive']['Wied99'] = {'cc0': 3.28,
                                     'cc1': 1,
                                     'cc2': 5.44,
                                     'cc3': -8,
                                     'cc4': -0.35,
                                     'cc5': 0.35,
                                     'cc6': 0,
                                     'cc7': 1.08,
                                     'cc8': 12.46,
                                     'cc9': 5.9}

# Krauss
CFmodel['Human']['Krauss'] = {'minGap': 2.5,
                              'accel': 2.6,
                              'decel': 4.5,
                              'sigma': 0.5,
                              'tau': 1,
                              'emergencyDecel': 9}

CFmodel['AVnormal']['Krauss'] = {'minGap': 0.5,
                                 'accel': 3.8,
                                 'decel': 4.5,
                                 'sigma': 0,
                                 'tau': 0.4,
                                 'emergencyDecel': 9}

CFmodel['AVsafe']['Krauss'] = {'minGap': 0.5,
                               'accel': 3.8,
                               'decel': 4.5,
                               'sigma': 0,
                               'tau': 0.4,
                               'emergencyDecel': 9}

CFmodel['AVaggressive']['Krauss'] = {'minGap': 0.5,
                                     'accel': 3.8,
                                     'decel': 4.5,
                                     'sigma': 0,
                                     'tau': 0.4,
                                     'emergencyDecel': 9}

# IDM
CFmodel['Human']['IDM'] = {'minGap': 2.5,
                           'accel': 2.6,
                           'decel': 4.5,
                           'tau': 1,
                           'emergencyDecel': 9,
                           'delta': 4}

CFmodel['AVnormal']['IDM'] = {'minGap': 1.2,
                              'accel': 1.2,
                              'decel': 4.5,
                              'tau': 1,
                              'emergencyDecel': 9,
                              'delta': 4}

CFmodel['AVsafe']['IDM'] = {'minGap': 1.2,
                            'accel': 1.2,
                            'decel': 4.5,
                            'tau': 1,
                            'emergencyDecel': 9,
                            'delta': 4}

CFmodel['AVaggressive']['IDM'] = {'minGap': 1.2,
                                  'accel': 1.2,
                                  'decel': 4.5,
                                  'tau': 1,
                                  'emergencyDecel': 9,
                                  'delta': 4}
# CoopVisW99
CFmodel['Human']['CoopVisWied99'] = {
    'minLookAheadDist': 0,
    'maxLookAheadDist': 820.21,
    'minLookBackDist': 0,
    'maxLookBackDist': 492.13,
    'noOfInteractObjects': 2}

CFmodel['AVnormal']['CoopVisWied99'] = {
    'minLookAheadDist': 0,
    'maxLookAheadDist': 984.25,
    'minLookBackDist': 0,
    'maxLookBackDist': 656.16,
    'noOfInteractObjects': 10}

CFmodel['AVsafe']['CoopVisWied99'] = {
    'minLookAheadDist': 0,
    'maxLookAheadDist': 984.25,
    'minLookBackDist': 0,
    'maxLookBackDist': 656.16,
    'noOfInteractObjects': 10}

CFmodel['AVaggressive']['CoopVisWied99'] = {
    'minLookAheadDist': 0,
    'maxLookAheadDist': 984.25,
    'minLookBackDist': 0,
    'maxLookBackDist': 656.16,
    'noOfInteractObjects': 10}

# LCVisWied99
CFmodel['Human']['LCVisWied99'] = {
    'safetyDistRedFact': 0.60,
    'coopLaneChange': False,
    'minClearance': 1.64,
}

CFmodel['AVnormal']['LCVisWied99'] = {
    'safetyDistRedFact': 0.45,
    'coopLaneChange': True,
    'minClearance': 1.64,
}

CFmodel['AVsafe']['LCVisWied99'] = {
    'safetyDistRedFact': 0.45,
    'coopLaneChange': True,
    'minClearance': 1.64,
}

CFmodel['AVaggressive']['LCVisWied99'] = {
    'safetyDistRedFact': 0.45,
    'coopLaneChange': True,
    'minClearance': 1.64,
}

# LCSumoKrauss
CFmodel['Human']['LCSumoKrauss'] = {
    'LcStrategic': 1,
    'LcCooperative': 1,
    'LcAccelLat': 1
}

CFmodel['AVnormal']['LCSumoKrauss'] = {
    'LcStrategic': 1,
    'LcCooperative': 1,
    'LcAccelLat': 1
}

CFmodel['AVaggressive']['LCSumoKrauss'] = {
    'LcStrategic': 1,
    'LcCooperative': 1,
    'LcAccelLat': 1
}

CFmodel['AVsafe']['LCSumoKrauss'] = {
    'LcStrategic': 1,
    'LcCooperative': 1,
    'LcAccelLat': 1
}

# LCSumoIDM
CFmodel['Human']['LCSumoIDM'] = {
    'LcStrategic': 1,
    'LcCooperative': 1
}

CFmodel['AVnormal']['LCSumoIDM'] = {
    'LcStrategic': 1,
    'LcCooperative': 1
}

CFmodel['AVsafe']['LCSumoIDM'] = {
    'LcStrategic': 1,
    'LcCooperative': 1
}

CFmodel['AVaggressive']['LCSumoIDM'] = {
    'LcStrategic': 1,
    'LcCooperative': 1
}
