import numpy as np
import sys

"""
Utilities file, includes helper functions and measures
"""

def segregation_index(model, unit = "school" ):

    """
    Calculates the local compositions for schools/neighbourhoods and then the segregation
    @unit: "school", "neighbourhood" or "agents_neighbourhood"

    """
    pi_jm = np.zeros(shape=(len(model.school_locations), len(model.household_types)))
    local_compositions = np.zeros(shape=(len(model.school_locations), len(model.household_types)))


    if unit == "school":

        for s_ind, school in enumerate(model.schools):
            local_composition = school.get_local_composition()

            local_compositions[s_ind][:] = local_composition
            pi_jm[s_ind][:] = local_composition / np.sum(local_composition)


    elif unit == "neighbourhood":
        for s_ind, school in enumerate(model.schools):
            local_composition = school.get_local_neighbourhood_composition()
            local_compositions[s_ind][:] = local_composition
            pi_jm[s_ind][:] = local_composition / np.sum(local_composition)


    elif unit == "agents_neighbourhood":

        pi_jm = np.zeros(shape=(len(model.households), len(model.household_types)))
        local_compositions = np.zeros(shape=(len(model.households), len(model.household_types)))

        for a_ind, household_agent in enumerate(model.households):
            local_composition = household_agent.get_local_neighbourhood_composition()
            local_compositions[a_ind][:] = local_composition
            pi_jm[a_ind][:] = local_composition / np.sum(local_composition)



    else:
        print("Not valid segregation measure")
        sys.exit()




    seg_index = calculate_segregation_index(local_compositions,pi_jm)
    if seg_index:
        return(seg_index)
    else:
        return(np.nan)


def calculate_segregation_index(local_compositions, pi_jm):

    """
    :param model

    pi_jm: proportions in unit j that belongs to group m, shape: (j,m) (schools,groups)
    pm: proportion in group m (m,1)

    tj: total number in group j - dim: (1,j)

    :return: information theory index

    """

    T=np.sum(local_compositions)

    tj = np.sum(local_compositions,axis=1, keepdims=True)

    #pm = np.sum(pi_jm,axis=0)/np.sum(pi_jm, keepdims=True)
    pm = np.sum(local_compositions,axis=0)/np.sum(local_compositions, keepdims=True)

    #print("pm",pm)

    E = np.sum(pm*np.log(1/pm))

    #print("tj/TE",tj / (T * E))
    #print("pi_jm",pi_jm)
    #print("pm",pm)

    log_matrix= np.nan_to_num(np.log(pi_jm/pm))
    print('log',log_matrix)

    seg_index = np.sum(tj / (T*E) * pi_jm * log_matrix, axis=None)

    print("pm", pm)

    if seg_index>1:
        print("tj",tj )

        print("tj/TE",tj / (T * E))
        print("pi_jm",pi_jm)
        print("pi_jm/pm",pi_jm/pm)

        print("pm",pm)

    return(seg_index)


def dissimilarity_index(model):

    for s_ind, school in enumerate(model.schools):
        local_composition = school.get_local_composition()
        model.pi_jm[s_ind][:] = local_composition

    pi_jm = model.pi_jm
    T=np.sum(pi_jm)
    tj = np.sum(pi_jm,axis=0, keepdims=True)
    pm = np.sum(pi_jm, axis=1, keepdims=True)

