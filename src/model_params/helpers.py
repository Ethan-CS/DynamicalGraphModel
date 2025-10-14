from enum import Enum


class Coupling(Enum):
    NEIGHBOUR_ENTER = 0
    NEIGHBOUR_EXIT = 1
    ISOLATED_ENTER = 2
    ISOLATED_EXIT = 3


def parse_coupling_descriptor(descriptor):
    """Parse a coupling descriptor into its component states.

    Returns a tuple ``(s1, s2, s3, s4)`` representing the states involved in
    the driving interaction (``s1`` and optional neighbour ``s2``) together with
    the source (``s3``) and destination (``s4``) states of the transition. When
    the descriptor omits any of these components they are inferred using the
    same defaults as :meth:`CModel.set_coupling_rate`.
    """

    transition_part, _, remainder = descriptor.partition(':')
    transition_part = transition_part.strip()
    remainder = remainder.strip()

    lhs_states = [token for token in transition_part.split('*') if token]
    s1 = lhs_states[0] if lhs_states else None
    s2 = lhs_states[1] if len(lhs_states) > 1 else None

    s3 = s4 = None
    if remainder:
        if '=>' in remainder:
            src_part, dst_part = remainder.split('=>', 1)
            src_part = src_part.strip()
            dst_part = dst_part.strip()
            s3 = src_part or None
            s4 = dst_part or None
        else:
            s4 = remainder or None

    if s3 is None and s4 is None:
        # No explicit target provided; mirror the driving states.
        if s1 is not None and s2 is not None:
            s3, s4 = s1, s2
        elif s1 is not None:
            s3 = s4 = s1
    elif s3 is None and s4 is not None:
        # Target provided but no explicit source; default to the first driver.
        s3 = s1

    return s1, s2, s3, s4


def coupling_types(model):
    _coupling_map = {state: [] for state in model.states}

    for couple, (descriptor, rate) in model.couplings.items():
        s1, s2, s3, s4 = parse_coupling_descriptor(descriptor)
        uses_neighbour = s2 is not None

        if s3 in _coupling_map:
            exit_type = Coupling.NEIGHBOUR_EXIT if uses_neighbour else Coupling.ISOLATED_EXIT
            _coupling_map[s3].append((exit_type, descriptor, rate, couple))

        if s4 in _coupling_map:
            entry_type = Coupling.NEIGHBOUR_ENTER if uses_neighbour else Coupling.ISOLATED_ENTER
            _coupling_map[s4].append((entry_type, descriptor, rate, couple))

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

