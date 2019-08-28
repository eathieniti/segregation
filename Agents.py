from mesa import Model, Agent
import mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid, SingleGrid
from mesa.datacollection import DataCollector
from scipy.spatial import distance, Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import random
from collections import Counter
from util import segregation_index, calculate_segregation_index, dissimilarity_index, \
    calculate_collective_utility, get_counts_util

print("mesa",mesa.__file__)


class SchoolAgent(Agent):
    '''

    '''
    def __init__(self, pos, model):

        super().__init__(pos, model)

        self.students = []
        self.type = 2
        self.pos = pos



        # measures
        self.local_composition = [0,0]
        # TODO: households for now: change to students
        self.capacity = 0
        self.neighbourhood_students = []

    def step(self):
        pass


    def get_local_school_composition(self):

        # get the composition of the students in the neighbourhood

        local_composition = get_counts_util(students=self.students, model=self.model)

        self.local_composition = local_composition

        self.current_capacity = np.sum(self.local_composition)

        return(local_composition)


    def get_local_neighbourhood_composition(self):

        """
        get the composition of the students in the neighbourhood

        :return: [number of type 0, number of type 1]
        """

        local_neighbourhood_composition = get_counts_util(self.neighbourhood_students, self.model)
        #print("step ",self.model.schedule.steps," neighb students ",len(self.neighbourhood_students))

        return (local_neighbourhood_composition)





class HouseholdAgent(Agent):
    '''
    Schelling segregation agent
    '''
    def __init__(self, pos, model, agent_type):
        '''
         Create a new Schelling agent.

         Args:
            unique_id: Unique identifier for the agent.
            x, y: Agent initial location.
            agent_type: Indicator for the agent's type (minority=1, majority=0)
        '''
        super().__init__(pos, model)
        self.type = agent_type


        # draw a value for f from a normal distribution
        if self.model.variable_f:
            self.f = np.random.normal(model.f[agent_type], model.sigma)
        else:
            self.f = model.f[agent_type]


        # TODO: extend fs to allow different numbers for different agent types
        if model.fs != "eq":
            self.fs = np.random.normal(model.fs, model.sigma)
        else:
            self.fs = self.f


        self.M = model.M[agent_type]




        self.T = model.T
        self.children = 1
        self.school = None
        self.dist_to_school = None
        self.local_composition = None
        self.fixed_local_composition = None
        self.variable_local_composition = None

        self.pos = pos

        self.closer_school = 0
        self.Dj = []
        self.schelling = self.model.schelling
        self.school_utilities = []
        self.residential_utilities = []




    # def calculate_distances(self):
    #     '''
    #     calculate distance between school and household
    #     Euclidean or gis shortest road route
    #     :return: dist
    #     '''
    #     Dj = np.zeros((len(self.model.school_locations),1))
    #
    #     for i, loc in enumerate(self.model.school_locations):
    #
    #         Dj[i] = np.linalg.norm(np.array(self.pos)- np.array(loc))
    #     self.Dj = Dj
    #
    #     print("calculating distances", Dj)
    #
    #     closer_school_index = np.argmin(self.Dj)
    #     self.closer_school = self.model.schools[closer_school_index]
    #

    def calculate_distances(self, Dij):
        '''
        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist
        '''
        Dj = np.zeros((len(self.model.school_locations),1))

        for i, loc in enumerate(self.model.school_locations):

            Dj[i] = np.linalg.norm(np.array(self.pos)- np.array(loc))
        self.Dj = Dj

        print("calculating distances", Dj)

        closer_school_index = np.argmin(self.Dj)
        self.closer_school = self.model.schools[closer_school_index]


    def get_closer_school(self):

        return(self.closer_school)




    def step(self):
        if self.model.schedule.steps < self.model.residential_steps:
            residential_move = True
        else:
            residential_move = False


        if residential_move:
            # only step the agents if the number considered is not exhausted
            if self.model.total_considered < self.model.residential_moves_per_step:
                # move residential
                U_res = self.get_res_satisfaction(self.pos)
                self.model.res_satisfaction.append(U_res)

                # print("U_res",U_res)
                if U_res < self.T:

                    # todo: implement different move schemes, for now only random
                    # find all empty places
                    # rank them
                    # take one with boltzmann probability.
                    self.evaluate_move(U_res, school=False)

                else:
                    self.model.res_happy += 1

                self.model.total_considered += 1


        else:
            if self.model.total_considered < self.model.school_moves_per_step:
            # school moves
                # satisfaction in current school
                U = self.get_school_satisfaction(self.school, self.dist_to_school)
                self.model.satisfaction.append(U)

            # If unhappy, compared to threshold move:
                if U < self.T:
                    #print('unhappy')
                    self.evaluate_move(U, school=True)

                else:
                    self.model.happy += 1
                    if self.model.total_considered>0:
                        self.model.percent_happy = np.ma(self.model.happy/self.model.total_considered)




    def get_res_satisfaction(self, position):


        x, y = self.get_like_neighbourhood_composition(position, radius=self.model.radius ,bounded=self.model.bounded)

        p = x + y
        P = self.ethnic_utility(x=x, p=p, f=self.f, schelling=self.schelling)


        self.res_satisfaction = P

        return(P)


    def get_like_neighbourhood_composition(self, position, radius, bounded=False):

        # warning: for now only suitable for 2 gropups
        x = 0 # like neighbours
        y = 0 # unlike neighbours

        if bounded:
            composition = self.model.get_closer_school_from_position(position).get_local_neighbourhood_composition()
            #print("school comp ", composition)
            for type in [0,1]:
                if type == self.type:
                    x = composition[type]
                else:
                    y = composition[type]




        else:

            neighbours = self.model.grid.get_neighbors(position, moore=True, radius=radius)
            for neighbour in neighbours:
                if isinstance(neighbour, HouseholdAgent):
                    if neighbour.type == self.type:
                        x += 1
                    else:
                        y += 1
            #print('x,y,',x,y)

        return(x, y)



    def get_local_neighbourhood_composition(self, position, radius, bounded=False):

        # warning: for now only suitable for 2 gropups
        type1 = 0
        type2 = 0
        # bounded not working yet
        # local_composition[agent_type]
        # if bounded:
        #     x, y = self.get_closer_school().get_local_neighbourhood_composition()
        #
        # else:



        neighbours = self.model.grid.get_neighbors(position, moore=True, radius=radius)

        local_composition = get_counts_util(neighbours, self.model)


        return (local_composition)


    def allocate(self, school, dist):

        self.school = school
        school.students.append(self)
        self.dist_to_school = dist



    def evaluate_move(self,U, school=True):

        if school:
            # choose proportional or deterministic
            utilities = self.get_school_utilities()
            index_to_move = self.choose_candidate(U,utilities)
            self.move_school(index_to_move, self.model.schools[index_to_move])
            self.school_utilities = utilities

            #print("utilities",utilities)

        else:
            residential_candidates, utilities = self.get_residential_utilities()
            index_to_move = self.choose_candidate(U,utilities)
            self.move_residence(residential_candidates[index_to_move], bounded=self.model.bounded)
            self.residential_utilities = utilities
            #print("utilities",utilities)




    def choose_candidate(self, U, utilities):

        boltzmann_probs = []
        proportional_probs = []

        if self.model.move == "deterministic":
            proportional_probs = utilities / np.sum(utilities)

            index_to_move = np.argmax(np.array(proportional_probs))


        if self.model.move == "proportional":


            proportional_probs = utilities / np.sum(utilities)
            index_to_move = np.random.choice(len(proportional_probs), p=proportional_probs)


        elif self.model.move == "random":

            index_to_move = random.randint(0, len(utilities)-1)


        elif self.model.move == "boltzmann":
            for U_candidate in utilities:
                boltzmann_probs.append(self.get_boltzman_probability(U, U_candidate))
            # normalize probailities to sum to 1
            boltzmann_probs_normalized = boltzmann_probs / np.sum(boltzmann_probs)
            index_to_move = np.random.choice(len(boltzmann_probs_normalized), p=boltzmann_probs_normalized)
        else:
            print("No valid move recipe selected")

        return(index_to_move)

    def get_school_utilities(self):

        utilities = []
        for school_index, candidate_school in enumerate(self.model.schools):
            # check whether school is eligible to move to
            # if candidate_school.current_capacity <= (candidate_school.capacity + 10) and candidate_school!=self.school:
            #print(candidate_school.current_capacity,candidate_school.capacity )
            if candidate_school.current_capacity <= (candidate_school.capacity):
                U_candidate = self.get_school_satisfaction(candidate_school, dist=self.Dj[school_index])
                utilities.append(U_candidate)

            else:
                utilities.append(0)
                # print(self.pos, candidate_school.pos, candidate_school.unique_id,self.Dj[school_index] )
        if len(utilities) != len(self.model.schools):

            print("Error: not all schools are being evaluated")
            sys.exit()
        return utilities

    def get_residential_utilities(self):

        empties = [] # just a list of positions
        candidates = [] # just a list of positions
        utilities = [] # utilities of the candidate positions

        # Evaluate all residential sites
        empties = self.model.grid.empties
        # convert set to list
        empties = list(empties)


        # just to make things faster only consider a subset of empty sites
        random.shuffle(empties)
        empties_shuffled = empties[0::self.model.sample]

        #empties_shuffled =empties
        for e in empties_shuffled:
            #if e not in candidates and self.model.grid.is_cell_empty(e):
            if e not in candidates:

                # TODO: empty site find the closer school
                U_res_candidate = self.get_res_satisfaction(e)
                #print("cand,util",e, U_res_candidate)
                utilities.append(U_res_candidate)
                candidates.append(e)

        # also add the current position
        candidates.append(self.pos)

        utilities.append(self.get_res_satisfaction(self.pos))
        return candidates, utilities

    def get_boltzman_probability(self,U, U_candidate):

        deltaU = U_candidate-U

        return(1/(1+np.exp(-deltaU/self.model.temp)))




    def move_school(self, school_index, new_school):

        # Removes student from current school and allocates to new
        # only do the actually move if it is really a different school otherwise stay
        if self.model.schools[school_index] != self.school:


            self.school.students.remove(self)

            # update metrics for school - could be replaced by +-1
            self.school.get_local_school_composition()

            # allocate elsewhere
            self.allocate(new_school, self.Dj[school_index])

            # now update the new school
            self.school.get_local_school_composition()

            self.model.total_moves +=1



    def move_residence(self, new_position, bounded=False):
        """


        :param new_position: (x,y) location to move to
        :return: None
        """

        if bounded:

            self.closer_school.neighbourhood_students.remove(self)

            self.model.grid.move_agent(self, new_position)

            new_school = self.model.get_closer_school_from_position(new_position)

            new_school.neighbourhood_students.append(self)


            self.model.res_moves += 1

        else:

            self.model.grid.move_agent(self, new_position)

            self.model.res_moves += 1



    def get_school_satisfaction(self, school, dist):
        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # For the schools we add the distance satisfaction


        x = school.get_local_school_composition()[self.type]
        p = np.sum(school.get_local_school_composition())

        dist = float(dist)

        P = self.ethnic_utility(x=x,p=p, f=self.fs,schelling =self.schelling)


        D = (self.model.max_dist - dist) / self.model.max_dist
        #print("D", D)
        U = P**(self.model.alpha) * D**(1-self.model.alpha)
        #print("P,D,U",P,D,U)

        return(U)


    def ethnic_utility(self, x, p, f, schelling=False):

        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # satisfaction
        fp = float(f * p)
        #print("fp,x",fp,x)


        # print(x,p)
        P = 0

        if self.type in [0, 1]:

            if schelling:
                if x <= fp:
                    P=0
                else:
                    P=1

            else:
                if fp == 0:
                    P = 0
                elif x <= fp:
                    P = float(x) / fp

                else:
                    P = self.M + (p - x) * (1 - self.M) / (p * (1 - f))



        return(P)




