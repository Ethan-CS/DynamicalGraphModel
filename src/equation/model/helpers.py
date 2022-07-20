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
