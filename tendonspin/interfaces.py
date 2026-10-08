# Source: packaged Boya actuator model; user authorizes all five fingers while
# holding the wrist. This named interface is not a hardware calibration.
"""Current five-finger action layout; no simulator dependency."""
from copy import deepcopy

FINGERS = ('TH', 'FF', 'MF', 'RF', 'LF')
ACTION_NAMES = tuple(f + 'J' + str(j) for f in FINGERS
                     for j in ((4, 3, 2, 1) if f == 'TH' else (4, 3, 2)))
ACTION_DIM = len(ACTION_NAMES)
ACTION_LAYOUT = 'boya_fingers16_wrist_hold_v1'
HELD_MOTOR_NAMES = ('FAJ3', 'FAJ1')
PASSIVE_JOINT_NAMES = ('FFJ1', 'MFJ1', 'RFJ1', 'LFJ1')
LEGACY_ACTION_NAMES = ACTION_NAMES[:13]
PROPRIO_FRAME_DIM = 2 * ACTION_DIM
STUDENT_SCHEMA = 'motor16_padforce15_action16_history20_v2'
STUDENT_FRAME_DIM = 3 * ACTION_DIM + 3 * len(FINGERS)
STUDENT_HISTORY = 20


def action_motor_indices(contract, action_names=ACTION_NAMES):
    """Resolve actual named motors; wrists and passive links are not policy actions."""
    names = tuple(action_names)
    if not names or len(names) != len(set(names)) or not set(names).issubset(ACTION_NAMES):
        raise ValueError('Unique active finger motor names required')
    motor_names = [a['name'] for a in contract['actuators']]
    return tuple(motor_names.index(name) for name in names)


def finger_action_contract(source):
    """Retain the original physical model/state, explicitly open all finger motors."""
    action_motor_indices(source)
    result = deepcopy(source)
    result.update(action_names=list(ACTION_NAMES), action_layout=ACTION_LAYOUT,
                  held_motor_names=list(HELD_MOTOR_NAMES),
                  passive_joint_names=list(PASSIVE_JOINT_NAMES))
    return result
