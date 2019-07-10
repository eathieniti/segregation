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

        # get the composition of the students in the neighbourhood

        local_neighbourhood_composition = get_counts_util(self.neighbourhood_students, self.model)

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
        self.f = model.f[agent_type]
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





    def calculate_distances(self):
        '''
        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist
        '''
        Dj = np.zeros((len(self.model.school_locations),1))

        for i, loc in enumerate(self.model.school_locations):

            Dj[i] = np.linalg.norm(np.array(self.pos)- np.array(loc))
        self.Dj = Dj

        #print("calculating distances", Dj)

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
        P = self.ethnic_utility(x=x, p=p, schelling=self.schelling)


        self.res_satisfaction = P

        return(P)


    def get_like_neighbourhood_composition(self, position, radius, bounded=False):

        # warning: for now only suitable for 2 gropups
        x = 0 # like neighbours
        y = 0 # unlike neighbours

        if bounded:
            x, y = self.get_closer_school().get_local_neighbourhood_composition()

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

        else:
            residential_candidates, utilities = self.get_residential_utilities()

            index_to_move = self.choose_candidate(U,utilities)
            self.move_residence(residential_candidates[index_to_move])


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
            if candidate_school.current_capacity <= (candidate_school.capacity):
                U_candidate = self.get_school_satisfaction(candidate_school, dist=self.Dj[school_index])
                utilities.append(U_candidate)

            else:
                utilities.append(0)
                # print(self.pos, candidate_school.pos, candidate_school.unique_id,self.Dj[school_index] )

        if len(utilities) != len(self.model.schools):

            print("Error: not all schools are being evaluated")
        return utilities

    def get_residential_utilities(self):

        utilities = []
        candidates = []
        empties = []
        # Evaluate all residential sites
        empties = self.model.grid.empties

        # just to make things faster..
        #empties_shuffled = empties[0::5]
        #random.shuffle(empties_shuffled)

        empties_shuffled =empties
        for e in empties_shuffled:
            if e not in candidates and self.model.grid.is_cell_empty(e):
                # TODO: empty site find the closer school
                U_res_candidate = self.get_res_satisfaction(e)
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



    def move_residence(self, new_position):
        self.model.grid.move_agent(self, new_position)

        self.model.res_moves += 1




    #@property
    def get_school_satisfaction(self, school, dist):
        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # For the schools we add the distance satisfaction


        x = school.get_local_school_composition()[self.type]
        p = np.sum(school.get_local_school_composition())

        dist = float(dist)

        P = self.ethnic_utility(x,p, schelling =self.schelling)


        D = (self.model.max_dist - dist) / self.model.max_dist
        #print("D", D)
        U = P**(self.model.alpha) * D**(1-self.model.alpha)
        #print("P,D,U",P,D,U)

        return(U)


    def ethnic_utility(self, x, p, schelling=False):

        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # satisfaction
        fp = float(self.f * p)
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
                    P = self.M + (p - x) * (1 - self.M) / (p * (1 - self.f))



        return(P)






class SchoolModel(Model):
    '''
    Model class for the Schelling segregation model.
    '''

    def __init__(self, height=100, width=100, density=0.95, num_schools=64,minority_pc=0.5, homophily=3, f0=0.6,f1=0.6,\
                 M0=0.8,M1=0.8,T=0.75,
                 alpha=0.2, temp=0.1, cap_max=1.01, move="boltzmann", symmetric_positions=True,
                 residential_steps=200,schelling=False,bounded=False,
                 residential_moves_per_step=500, school_moves_per_step = 500,radius=7,proportional = False,
                 torus=False):
        '''
        '''
        # Options  for the model
        self.height = height
        self.width = width
        print("h x w",height, width)
        self.density = density
        self.num_schools= num_schools
        self.homophily = homophily
        self.f = [f0,f1]
        self.M = [M0,M1]
        self.residential_steps = residential_steps
        self.minority_pc = minority_pc
        self.bounded = bounded
        self.cap_max=cap_max
        self.T = T
        self.radius = radius
        self.household_types = [0, 1] # majority, minority !!
        self.symmetric_positions = symmetric_positions
        self.schelling=schelling


        # choice parameters
        self.alpha = alpha
        self.temp = temp

        self.households = []
        self.schools = []
        self.residential_moves_per_step = residential_moves_per_step
        self.school_moves_per_step = school_moves_per_step


        self.num_households = int(width*height*density)
        num_min_households = int(self.minority_pc * self.num_households)
        self.pm = [self.num_households-num_min_households, num_min_households]

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=torus)
        self.total_moves = 0
        self.res_moves = 0

        self.move = move

        self.school_locations = []
        self.household_locations = []
        self.Dij = []


        self.happy = 0
        self.res_happy = 0
        self.percent_happy = 0
        self.seg_index = 0
        self.res_seg_index = 0
        self.residential_segregation = 0
        self.collective_utility = 0
        self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7, \
        self.comp8, self.comp9, self.comp10, self.comp11, self.comp12, self.comp13, self.comp14, self.comp15 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        self.satisfaction = []
        self.pi_jm = []
        self.pi_jm_fixed = []
        self.compositions = []
        self.average_like_fixed = 0
        self.average_like_variable = 0


        self.my_collector = []
        if torus:
            self.max_dist = self.height/np.sqrt(2)
        else:
            self.max_dist = self.height*np.sqrt(2)



        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        # Set up schools in symmetric positions along the grid

        if num_schools == 4:
            school_positions = [(width/4,height/4),(width*3/4,height/4),(width/4,height*3/4),(width*3/4,height*3/4)]
        elif num_schools == 9:
            n=6
            school_positions = [(width/n,height/n),(width*3/n,height*1/n),(width*5/n,height*1/n),(width/n,height*3/n),\
                                (width*3/n,height*3/n),(width*5/n,height*3/n),(width*1/n,height*5/n),(width*3/n,height*5/n),\
                                (width*5/n,height*5/n)]
        elif num_schools == 16:
            school_positions = []
            n=8
            x1 = [1, 3, 5, 7]

            xloc = np.repeat(x1, 4)
            yloc = np.tile(x1, 4)

            for i in range(len(x1 * 4)):
                school_positions.append((xloc[i] * height / n, yloc[i] * width / n))

        elif num_schools == 64:
            school_positions = []
            n=int(np.sqrt(num_schools)*2)
            print(n)
            x1 = range(1,int(n+1),2)

            xloc = np.repeat(x1, int(n/2))
            yloc = np.tile(x1, int(n/2))

            for i in range(num_schools):
                school_positions.append((xloc[i] * height / n, yloc[i] * width / n))



        for i in range(self.num_schools):
            #Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)

            if self.symmetric_positions:
                pos = (int(school_positions[i][0]),int(school_positions[i][1]))
            else:
                pos = (x,y)
            self.school_locations.append(pos)
            school = SchoolAgent(pos, self)
            self.grid.place_agent(school, school.unique_id)
            self.schools.append(school)
            self.schedule.add(school)

        # Set up households


        # create household locations but dont create agents yet


        while len(self.household_locations) < self.num_households:

            #Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            pos = (x,y)

            if (pos not in (self.school_locations) ) and (pos not in self.household_locations):
                self.household_locations.append(pos)






        #print(Dij)

        for ind, pos in enumerate(self.household_locations):
            # create a school or create a household

            if ind < int(self.minority_pc*self.num_households):
                agent_type = self.household_types[1]
            else:
                agent_type = self.household_types[0]


            agent = HouseholdAgent(pos, self, agent_type)
            #decorator_agent = HouseholdAgent(pos, self, agent_type)

            self.grid.place_agent(agent, agent.unique_id)

            #self.grid.place_agent(decorator_agent, pos)


            agent.calculate_distances()

            self.households.append(agent)

            random_school_index = random.randint(0, len(self.schools)-1)
            candidate_school = self.schools[random_school_index]
            agent.allocate(candidate_school,agent.Dj[random_school_index])

            # closer_school = household.schools[np.argmin(household.)]



            #closer_school = self.schools[p.argmin(Dj)]
            #closer_school.students.append(agent)
           # agent.allocate(closer_school, np.min(Dj))
            #print(agent.school.unique_id)



            self.schedule.add(agent)

        self.pi_jm = np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.local_compositions =  np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.avg_school_size = round(density*width*height/(len(self.schools)))

        self.datacollector = DataCollector(
            model_reporters={"agent_count":
                                 lambda m: m.schedule.get_agent_count(), "seg_index": "seg_index",
                             "residential_segregation": "residential_segregation", "res_seg_index":  "res_seg_index","fixed_res_seg_index":"fixed_res_seg_index",
                             "happy": "happy", "percent_happy": "percent_happy",
                             "total_moves": "total_moves", "compositions0": "compositions0",
                             "compositions1": "compositions1",
                                     "comp0": "comp0", "comp1": "comp1", "comp2": "comp2", "comp3": "comp3", "comp4": "comp4", "comp5": "comp5", "comp6": "comp6",
                             "comp7": "comp7","compositions": "compositions",
                             "collective_utility":"collective_utility",
                             "res_satisfaction": "res_satisfaction","satisfaction":"satisfaction"
                             },
            agent_reporters={"local_composition": "local_composition", "type": lambda a: a.type,
                             "id": lambda a: a.unique_id, "fixed_local_composition": "fixed_local_composition","variable_local_composition": "variable_local_composition"})


        # Calculate local composition
        # set size
        for school in self.schools:
            school.get_local_school_composition()
            #cap = round(np.random.normal(loc=cap_max * self.avg_school_size, scale=self.avg_school_size * 0.05))
            cap = self.avg_school_size * self.cap_max

            print("cap",self.avg_school_size, cap)
            segregation_index(self)
        #

        print("height = %d; width = %d; density = %.2f; num_schools = %d; minority_pc =  %.2f; homophily = %d; "
              "f0 =  %.2f; f1 =  %.2f; M0 =  %.2f; M1 =  %.2f;\
        alpha =  %.2f; temp =  %.2f; cap_max =  %.2f; move = %s; symmetric_positions = %s"%(height,
         width, density, num_schools,minority_pc,homophily,f0,f1, M0,M1,alpha,
                                       temp, cap_max, move, symmetric_positions ))

        self.total_considered = 0
        self.running = True
        self.datacollector.collect(self)




    # def calculate_distances(self):
    #     '''
    #     calculate distance between school and household
    #     Euclidean or gis shortest road route
    #     :return: dist
    #     '''
    #
    #     Dij = distance.cdist(np.array(self.household_locations), np.array(self.school_locations), 'euclidean')
    #     return(Dij)




    def step(self):
        '''
        Run one step of the model. If All agents are happy, halt the model.
        '''
        self.happy = 0  # Reset counter of happy agents
        self.res_happy = 0
        self.total_moves = 0
        self.total_considered = 0
        self.res_moves = 0
        self.satisfaction = []
        self.res_satisfaction=[]

        self.schedule.step()

        print("happy", self.happy)
        print("total_considered", self.total_considered)




        # Once residential steps are done calculate school distances



        if self.schedule.steps < self.residential_steps - 1 or self.schedule.steps ==1 :
            # during the residential steps keep recalculating the school neighbourhood compositions

            #print("recalculating neighbourhoods")
            for school in self.schools:
                school.neighbourhood_students = []



            household_locations = []


            for household in self.households:
                household.calculate_distances()
                # Calculate distances of the schools - define the school-neighbourhood and compare
                # closer_school = household.schools[np.argmin(household.)]
                closer_school_index = np.argmin(household.Dj)
                household.closer_school = self.schools[closer_school_index]
                household.closer_school.neighbourhood_students.append(household)

                # Initialize house allocation to school
                #household.move_school(closer_school_index, self.schools[closer_school_index])







        self.seg_index = segregation_index(self)
        self.residential_segregation = segregation_index(self, unit="neighbourhood")
        self.res_seg_index = segregation_index(self, unit="agents_neighbourhood")
        self.fixed_res_seg_index = segregation_index(self, unit="fixed_agents_neighbourhood", radius=1)
        satisfaction = np.mean(self.satisfaction)
        res_satisfaction = np.mean(self.res_satisfaction)



        print("seg_index", "%.2f"%(self.seg_index), "var_res_seg", "%.2f"%(self.res_seg_index), "neighbourhood",
              "%.2f"%(self.residential_segregation), "fixed_res_seg_index","%.2f"%(self.fixed_res_seg_index), \
              "res_satisfaction %.2f" %res_satisfaction,"satisfaction %.2f" %satisfaction,\
              "average_like_fixed %.2f"%self.average_like_fixed,"average_like_var %.2f"%self.average_like_variable  )

        # calculate these after residential_model
        if self.schedule.steps>self.residential_steps:
            self.collective_utility = calculate_collective_utility(self)
            print(self.collective_utility)



        if self.happy == self.schedule.get_agent_count():
            self.running = False
        compositions = []
        for school in self.schools:
            self.my_collector.append([self.schedule.steps, school.unique_id, school.get_local_school_composition()])
            self.compositions = school.get_local_school_composition()
            compositions.append(school.get_local_school_composition()[0])
            compositions.append(school.get_local_school_composition()[1])

            self.compositions1 = int(school.get_local_school_composition()[1])
            self.compositions0 = int(school.get_local_school_composition()[0])

        #print("comps",compositions,np.sum(compositions) )
        [self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7] = compositions[0:8]
        # collect data
        self.datacollector.collect(self)
        print("moves",self.total_moves, "res_moves", self.res_moves, "percent_happy", self.percent_happy)



