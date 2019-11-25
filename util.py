import numpy as np
import sys

"""
Utilities file, includes helper functions and measures
"""

def segregation_index(model, unit = "school" , radius=1):

    """
    Calculates the local compositions for schools/neighbourhoods and then the segregation
    @unit: "school", "neighbourhood" or "agents_neighbourhood"
    pi_jm = [proportion of like neighbours, proportion of unlike neighbours]

    """
    pi_jm = np.zeros(shape=(len(model.school_locations), len(model.household_types)))
    local_compositions = np.zeros(shape=(len(model.school_locations), len(model.household_types)))

    pm = [1-model.minority_pc, model.minority_pc]

    if unit == "school":
        all_s = []
        for s_ind, school in enumerate(model.schools):
            all_s.append(len(school.students))

        for s_ind, school in enumerate(model.schools):
            local_composition = school.get_local_school_composition()

            local_compositions[s_ind][:] = local_composition
            pi_jm[s_ind][:] = local_composition / np.sum(local_composition)

        #print("school compositions", local_compositions)

        # TODO: move to tests?
        print("total students in schools ",np.sum(local_compositions))

        if np.sum(local_compositions) != model.num_households:
            print("Error, not all agents counted in segregation index")


    # this only makes sense in the bounded model 
    elif unit == "neighbourhood":
        for s_ind, neighbourhood in enumerate(model.neighbourhoods):
            local_composition = neighbourhood.get_local_neighbourhood_composition()
            local_compositions[s_ind][:] = local_composition
            pi_jm[s_ind][:] = local_composition / np.sum(local_composition)
        #print("neigh compositions", local_compositions)
        print("total students in neighbourhood", np.sum(local_compositions))

        # TODO: move to tests?
        if np.sum(local_compositions) != model.num_households:
            print("Error, not all agents counted in segregation index")

    elif unit == "school_neighbourhood":
        for s_ind, school in enumerate(model.schools):
            local_composition = school.get_local_neighbourhood_composition()
            local_compositions[s_ind][:] = local_composition
            pi_jm[s_ind][:] = local_composition / np.sum(local_composition)
        #print("school neigh compositions", local_compositions)
        print("total students in schools ",np.sum(local_compositions))
        if np.sum(local_compositions) != model.num_households:
            print("Error, not all agents counted in segregation index")

    elif unit == "agents_neighbourhood":

        pi_jm = np.zeros(shape=(len(model.households), len(model.household_types)))
        local_compositions = np.zeros(shape=(len(model.households), len(model.household_types)))

        for a_ind, household_agent in enumerate(model.households):
            local_composition = household_agent.get_local_neighbourhood_composition(position=household_agent.pos,radius=model.radius)

            local_compositions[a_ind][:] = local_composition
            pi_jm[a_ind][:] = local_composition / np.sum(local_composition)
            model.pi_jm = pi_jm
            household_agent.variable_local_composition = local_composition

            #if model.schedule.steps>5:
            #    print(household_agent.pos,local_composition / np.sum(local_composition), local_composition, np.sum(local_composition))

            model.average_like_variable = np.mean(pi_jm[:,0])

    elif unit == "mixed":
        pi_jm = np.zeros(shape=(len(model.households), len(model.household_types)))
        local_compositions = np.zeros(shape=(len(model.households), len(model.household_types)))

        for a_ind, household_agent in enumerate(model.households):
            local_composition_variable = household_agent.get_local_neighbourhood_composition(position=household_agent.pos,
                                                                                    radius=model.radius)
            local_composition_bounded = household_agent.get_local_neighbourhood_composition(position=household_agent.pos,radius=None,bounded=True)

            local_composition = np.array(local_composition_bounded) * model.b_ef + np.array(local_composition_variable) * (1-model.b_ef)

            local_compositions[a_ind][:] = local_composition
            pi_jm[a_ind][:] = local_composition / np.sum(local_composition)
            model.pi_jm = pi_jm
            household_agent.variable_local_composition = local_composition


    elif unit == "fixed_agents_neighbourhood":

        pi_jm = np.zeros(shape=(len(model.households), len(model.household_types)))
        local_compositions = np.zeros(shape=(len(model.households), len(model.household_types)))

        for a_ind, household_agent in enumerate(model.households):

            local_composition = household_agent.get_local_neighbourhood_composition(position=household_agent.pos,
                                                                                    radius=radius)
            local_compositions[a_ind][:] = local_composition
            pi_jm[a_ind][:] = local_composition / np.sum(local_composition)
            model.pi_jm_fixed = pi_jm
            household_agent.fixed_local_composition = local_composition

            model.average_like_fixed = np.mean(pi_jm[:,0])


    else:
        print("Not valid segregation measure")
        sys.exit()




    seg_index = calculate_segregation_index(local_compositions,pi_jm, pm)
    if seg_index:
        return(seg_index)
    else:
        return(np.nan)


def calculate_segregation_index(local_compositions, pi_jm, pm):

    """
    Calculates Theil's segregation index

    :param local_compositions: (j,m)
        The numbers of each group for each unit
        eg. [[5,7]
            [7,5]]
    :param pi_jm: shape: (j,m) (schools,groups)
        proportions in unit j that belongs to group m
    :param pm: shape (m,1)
        proportion in group m
    :return: seg_index
        Theil's information theory index for segregation
    """

    #tj: total number in unit (school or neighbourhood) j - dim: (1,j)



    T=np.sum(local_compositions)
    tj = np.sum(local_compositions,axis=1, keepdims=True)
    #pm = np.sum(pi_jm,axis=0)/np.sum(pi_jm, keepdims=True)

    pm = np.array(pm)
    E = np.sum(pm*np.log(1/pm))

    #print("tj/TE",tj / (T * E))
    #print("pi_jm",pi_jm)
    #print("pm",pm)

    log_matrix= np.nan_to_num(np.log(pi_jm/pm))

    seg_index = np.sum(tj / (T*E) * pi_jm * log_matrix, axis=None)

    # print("pm", pm)

    return(seg_index)




def dissimilarity_index(model):

    for s_ind, school in enumerate(model.schools):
        local_composition = school.get_local_school_composition()
        model.pi_jm[s_ind][:] = local_composition

    pi_jm = model.pi_jm
    T=np.sum(pi_jm)
    tj = np.sum(pi_jm,axis=0, keepdims=True)
    pm = np.sum(pi_jm, axis=1, keepdims=True)




def calculate_collective_utility(model):
    """


    :param model:
    :return: average school satisfaction for all agents
    """
    utilities = []

    for household in  model.households:
        sat, p, d= household.get_school_satisfaction(household.school, household.dist_to_school)
        utilities.append(sat)

    print(np.mean(utilities))

    return(np.mean(utilities))


def calculate_res_collective_utility(model):
    """


    :param model:
    :return: average residential satisfaction for all agents
    """
    utilities = []

    for household in  model.households:
        sat = household.get_res_satisfaction(household.pos)
        utilities.append(sat)

    print(np.mean(utilities))

    return(np.mean(utilities))



def get_counts_util(students, model):
    """
    just gathers counts for each type, independent of agent's type
    works for schools, agents, and schools neighbourhoods
    needed for calculate_segregation_index() function

    :param students: list of HouseholdAgent objects
            A list of the students
    :param model:
    :return:
    """
    local_composition = [0, 0]

    d = [student.type for student in students]

    for agent_type in range(len(model.household_types)):
       local_composition[agent_type] = d.count(agent_type)

    return (local_composition)

