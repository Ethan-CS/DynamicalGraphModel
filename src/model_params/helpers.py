from enum import Enum


class Coupling(Enum):
    NEIGHBOUR_ENTER = 0
    NEIGHBOUR_EXIT = 1
    ISOLATED_ENTER = 2
    ISOLATED_EXIT = 3


# Maintain a global variable so that we do not
# need to re-determine map each time it is needed
_coupling_map = {}


def coupling_types(model):
    if _coupling_map == {}:
        # Initialise the dictionary keys (one for each model state)
        for state in model.states:
            _coupling_map[state] = []

        for state in model.states:
            for couple in model.couplings:
                # Get the situation under which a transition occurs
                transition = model.couplings[couple][0].split(':')[0]
                # Does the transition contain a state we are currently interested in?
                if model.couplings[couple][0].count(state) > 0:
                    if transition[0] == state:  # EXIT TRANSITION
                        if transition.count('*') > 0:  # NEEDS A NEIGHBOUR
                            _coupling_map[state].append((Coupling.NEIGHBOUR_EXIT, model.couplings[couple][0],
                                                         model.couplings[couple][1], couple))
                        else:
                            _coupling_map[state].append((Coupling.ISOLATED_EXIT, model.couplings[couple][0],
                                                         model.couplings[couple][1], couple))
                    else:  # ENTRY TRANSITION
                        if transition.count('*') > 0:  # NEEDS A NEIGHBOUR
                            _coupling_map[state].append((Coupling.NEIGHBOUR_ENTER, model.couplings[couple][0],
                                                         model.couplings[couple][1], couple))
                        else:
                            _coupling_map[state].append((Coupling.ISOLATED_ENTER, model.couplings[couple][0],
                                                        model.couplings[couple][1], couple))
    # TODO does this look at how you can end up (transition[x][-1]) in this state too?
    return _coupling_map


def dynamically_relevant(model):
    """
    Given a model, returns list of dynamically relevant model states.
    That is, all those states that are involved in active transition
    :type model: CModel
    """
    relevant = list()
    for state in model.states:
        for coupling in model.couplings:
            if model.couplings[coupling][0].split(':')[0].count(state) > 0:
                relevant.append(state)
                break
    return relevant

