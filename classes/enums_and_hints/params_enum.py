from enum import Enum

class DEMultStrategyNames(Enum):
    RAND1 = 'rand1'
    RANDTOBEST1 = 'randtobest1'

class GAParentSelectionNames(Enum):
    STEADY_STATE_SELECTION = 'sss'
    ROULETTE_WHEEL_SELECTION = 'rws'
    TOURNAMENT_SELECTION = 'tournament'

class GACrossoverNames(Enum):
    SINGLE_POINT = 'single_point'
    TWO_POINTS = 'two_points'

class GAMutationNames(Enum):
    RANDOM = 'random'
    SWAP = 'swap'
    ADAPTIVE = 'adaptive'
