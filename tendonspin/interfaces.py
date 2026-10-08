"""Hardware-facing actuator names; this module has no simulator dependency."""
FINGERS = ('TH','FF','MF','RF')
ACTION_NAMES = tuple(f + 'J' + str(j) for f in FINGERS for j in ((4,3,2,1) if f == 'TH' else (4,3,2)))
STUDENT_SCHEMA = 'motor13_padforce12_action13_history20_v1'
STUDENT_FRAME_DIM = 51
STUDENT_HISTORY = 20
