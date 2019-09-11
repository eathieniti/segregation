from mesa import Model, Agent
import mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid, SingleGrid
from mesa.datacollection import DataCollector
from scipy.spatial import distance, Voronoi, voronoi_plot_2d
import pandas as pd
import numpy as np
import random
import sys
from collections import Counter
from util import segregation_index, calculate_segregation_index, dissimilarity_index, \
    calculate_collective_utility, calculate_res_collective_utility, get_counts_util

from Agents import SchoolAgent, NeighbourhoodAgent, HouseholdAgent

print("mesa",mesa.__file__)




class SchoolModel(Model):


    """
    Model class for the Schelling segregation model.

    ...

    Attributes
    ----------

    height: int
        grid height
    width: int
        grid width
    num_schools:  int
        number of schools
    f : float
        fraction preference of agents for like
    M : float
        utility penalty for homogeneous neighbourhood
    residential_steps :
        number of steps for the residential model
    minority_pc :
        minority fraction
    bounded : boolean
        If True use bounded (predefined neighbourhood) for agents residential choice
    cap_max : float
        school capacity TODO: explain
    radius : int
        neighbourhood radius for agents calculation of residential choice (only used if bounded=False)
    household_types :
        labels for different ethnic types of households
    symmetric_positions :
        use regularly placed positions for the schools along the grid, or random
    schelling :
        if True use schelling utility function otherwise use assymetric
    school_pos :
        if supplied place schools in the supplied positions - also update school_num
    extended_data :
        if True collect extra data for agents (utility distribution and satisfaction)
        takes up a lot of space
    sample : int
        subsample the empty residential sites to be evaluated to speed up computation
    variable_f : variable_f
        draw values of the ethnic preference, f, from a normal distribution
    sigma : float
        The standard deviation of the normal distribution used for variable_f
    alpha : float
        ratio of ethnic to distance to school preference for school utility
    temp : float
        temperature for the behavioural logit rule for agents moving
    households : list
        all household objects
    schools : list
        all school objects
    residential_moves_per_step : int
        number of agents to move residence at every step
    school_moves_per_step : int
        number of agents to move school at every step
    num_households : int
        total number of household agents
    pm : list [ , ]
        number of majority households, number of minority households
    schedule : mesa schedule type
    grid : mesa grid type
    total_moves :
        number of school moves made in particular step
    res_moves :
        number of residential site moves made in particular step
    move :
        type of move recipe - 'random' 'boltzmann' or 'deterministic'
    school_locations : list
       list of locations of all schools (x,y)
    household_locations :
       list of locations of all households (x,y)
    closer_school_from_position : numpy array shape : (width x height)
        map of every grid position to the closest school

    """


    def __init__(self, height=100, width=100, density=0.9, num_neighbourhoods=16, schools_per_neighbourhood=2,minority_pc=0.5, f0=0.6,f1=0.6,\
                 M0=0.8,M1=0.8,T=0.75,
                 alpha=0.5, temp=1, cap_max=1.01, move="boltzmann", symmetric_positions=True,
                 residential_steps=70,schelling=False,bounded=False,
                 residential_moves_per_step=2000, school_moves_per_step =2000,radius=6,proportional = False,
                 torus=False,fs="eq", extended_data = False, school_pos=None, agents=None, sample=5, variable_f=True, sigma=0.1, displacement=4,
                 pow=1):


        # Options  for the model
        self.height = int(height)
        self.width = int(width)
        print("h x w",height, width)
        self.density = float(density)
        #self.num_schools= num_schools
        self.f = [float(f0),float(f1)]
        self.M = [M0,M1]
        self.residential_steps = int(residential_steps)
        self.minority_pc = minority_pc
        self.bounded = bounded
        self.cap_max=float(cap_max)
        self.T = T
        self.radius = radius
        self.household_types = [0, 1] # majority, minority, important !!
        self.symmetric_positions = symmetric_positions
        self.schelling=schelling
        self.school_pos = school_pos
        self.extended_data = extended_data
        self.sample = int(sample)
        self.variable_f = variable_f
        self.sigma = float(sigma)
        self.fs = fs
        self.pow = pow


        # choice parameters
        self.alpha = alpha
        self.temp = temp

        self.households = []
        self.schools = []
        self.neighbourhoods = []
        self.residential_moves_per_step = residential_moves_per_step
        self.school_moves_per_step = school_moves_per_step


        self.num_households = int(width*height*density)
        num_min_households = int(self.minority_pc * self.num_households)
        self.num_neighbourhoods = num_neighbourhoods
        self.schools_per_neigh = schools_per_neighbourhood
        self.num_schools = int(num_neighbourhoods * self.schools_per_neigh)
        self.pm = [self.num_households-num_min_households, num_min_households]

        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=torus)
        self.total_moves = 0
        self.res_moves = 0

        self.move = move

        self.school_locations = []
        self.household_locations = []
        self.neighbourhood_locations = []
        self.closer_school_from_position = np.empty([self.grid.width, self.grid.height])
        self.closer_neighbourhood_from_position = np.empty([self.grid.width, self.grid.height])


        self.happy = 0
        self.res_happy = 0
        self.percent_happy = 0
        self.seg_index = 0
        self.res_seg_index = 0
        self.residential_segregation = 0
        self.collective_utility = 0
        self.collective_res_utility = 0
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
        # TODO: the setting up of agents is messy... reorganize



        # if schools already supplied place them where they should be
        # TODO: fix
        if self.school_pos:
            school_positions = self.school_pos
            self.school_locations = school_pos
            self.num_schools = len(school_pos)
            print("Option not working")
            sys.exit()


        # otherwise calculate the positions
        else:
            if self.num_neighbourhoods == 4:
                neighbourhood_positions = [(width/4,height/4),(width*3/4,height/4),(width/4,height*3/4),(width*3/4,height*3/4)]
            elif self.num_neighbourhoods == 9:
                n=6
                neighbourhood_positions = [(width/n,height/n),(width*3/n,height*1/n),(width*5/n,height*1/n),(width/n,height*3/n),\
                                    (width*3/n,height*3/n),(width*5/n,height*3/n),(width*1/n,height*5/n),(width*3/n,height*5/n),\
                                    (width*5/n,height*5/n)]

            elif self.num_neighbourhoods in [25, 64, 16]:
                neighbourhood_positions = []
                n=int(np.sqrt( self.num_neighbourhoods)*2)
                print(n)
                x1 = range(1,int(n+1),2)

                xloc = np.repeat(x1, int(n/2))
                yloc = np.tile(x1, int(n/2))

                for i in range(self.num_neighbourhoods):
                    neighbourhood_positions.append((xloc[i] * height / n, yloc[i] * width / n))



        print(neighbourhood_positions)
        #for i in range(self.num_schools):i
        i=0
        while len(self.neighbourhoods)<self.num_neighbourhoods:
            n_school_pos=[0,0,0,0]
            if self.symmetric_positions or self.school_pos:
                x = int(neighbourhood_positions[i][0])
                y = int(neighbourhood_positions[i][1])

                #print(x,y)

            else:
                x = random.randrange(start=2,stop=self.grid.width-2)
                y = random.randrange(start=2,stop=self.grid.height-2)

            pos = (x,y)
            pos2 =(x+1,y+1)
            pos3 = (-100,-100)
            if schools_per_neighbourhood ==2:
                pos3 = (x-displacement,y-displacement)
                pos2 = (x+displacement,y+displacement)
            if schools_per_neighbourhood == 4:
                n_school_pos[0] = (x - displacement, y - displacement)
                n_school_pos[1] = (x + displacement, y + displacement)
                n_school_pos[2] = (x - displacement, y + displacement)
                n_school_pos[3] = (x + displacement, y - displacement)

            do_not_use = self.school_locations + self.neighbourhood_locations
            #if (pos not in do_not_use) and (pos2 not in do_not_use ) and (pos3 not in do_not_use ):
            if (pos not in do_not_use) and (pos2 not in do_not_use) and  (pos3 not in do_not_use ):

                #print('pos',pos,pos2,pos3)
                self.school_locations.append(pos2)
                school = SchoolAgent(pos2, self)
                self.grid.place_agent(school, school.unique_id)
                self.schools.append(school)
                self.schedule.add(school)

                if self.schools_per_neigh == 2:
                    # Add another school
                    self.school_locations.append(pos3)
                    school = SchoolAgent(pos3, self)
                    self.grid.place_agent(school, school.unique_id)
                    self.schools.append(school)
                    self.schedule.add(school)

                if self.schools_per_neigh == 4:
                    # Add another school
                    for si in range(0,self.schools_per_neigh):
                        self.school_locations.append( n_school_pos[si])
                        school = SchoolAgent(n_school_pos[si], self)
                        self.grid.place_agent(school, school.unique_id)
                        self.schools.append(school)
                        self.schedule.add(school)


                self.neighbourhood_locations.append(pos)
                neighbourhood = NeighbourhoodAgent(pos, self)
                self.grid.place_agent(neighbourhood, neighbourhood.unique_id)
                self.neighbourhoods.append(neighbourhood)
                self.schedule.add(neighbourhood)

            else:
                print(pos,pos2,pos3, "is found in",do_not_use )
            i+=1
        print("num_schools",len(self.school_locations))

        print("schools completed")

        #print(self.neighbourhood_locations)
        #print("schools",self.school_locations, len(self.school_locations))
        # Set up households

        # If agents are supplied place them where they need to be
        if agents:

            for cell in agents:
                [agent_type, x, y] = cell
                if agent_type in [0,1]:

                    pos = (x, y)
                    if self.grid.is_cell_empty(pos):
                        agent = HouseholdAgent(pos, self, agent_type)
                        self.grid.place_agent(agent, agent.unique_id)

                        self.household_locations.append(pos)
                        self.households.append(agent)
                        self.schedule.add(agent)


        # otherwise produce them
        else:

            # create household locations but dont create agents yet

            while len(self.household_locations) < self.num_households:

                #Add the agent to a random grid cell
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                pos = (x,y)

                if (pos not in (self.school_locations +  self.household_locations + self.neighbourhood_locations)):
                    self.household_locations.append(pos)



            #print(Dij)

            for ind, pos in enumerate(self.household_locations):

                # create a school or create a household

                if ind < int(self.minority_pc*self.num_households):
                    agent_type = self.household_types[1]
                else:
                    agent_type = self.household_types[0]

                household_index=ind
                agent = HouseholdAgent(pos, self, agent_type, household_index)
                #decorator_agent = HouseholdAgent(pos, self, agent_type)

                self.grid.place_agent(agent, agent.unique_id)

                #self.grid.place_agent(decorator_agent, pos)



                self.households.append(agent)
                self.schedule.add(agent)

        self.set_positions_to_school()
        self.set_positions_to_neighbourhood()
        self.calculate_all_distances_to_schools()
        self.calculate_all_distances_to_neighbourhoods()


        for agent in self.households:

            random_school_index = random.randint(0, len(self.schools)-1)
            #print("school_index", random_school_index, agent.Dj, len(agent.Dj))

            candidate_school = self.schools[random_school_index]
            agent.allocate(candidate_school,agent.Dj[random_school_index])



            #closer_school = self.schools[p.argmin(Dj)]
            #closer_school.students.append(agent)
           # agent.allocate(closer_school, np.min(Dj))
            #print(agent.school.unique_id)




        self.pi_jm = np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.local_compositions =  np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.avg_school_size = int(round(density*width*height/(len(self.schools))))

        if self.extended_data:
            self.datacollector = DataCollector(
                model_reporters={"agent_count":
                                     lambda m: m.schedule.get_agent_count(), "seg_index": "seg_index",
                                 "residential_segregation": "residential_segregation", "res_seg_index":  "res_seg_index","fixed_res_seg_index":"fixed_res_seg_index",
                                 "happy": "happy", "percent_happy": "percent_happy",
                                 "total_moves": "total_moves", "res_moves": "res_moves", "compositions0": "compositions0",
                                 "compositions1": "compositions1",
                                         "comp0": "comp0", "comp1": "comp1", "comp2": "comp2", "comp3": "comp3", "comp4": "comp4", "comp5": "comp5", "comp6": "comp6",
                                 "comp7": "comp7","compositions": "compositions",
                                 "collective_utility":"collective_utility", "collective_res_utility":"collective_res_utility"
                                 },
                agent_reporters={"local_composition": "local_composition", "type": lambda a: a.type,
                                 "id": lambda a: a.unique_id,
                                 #"fixed_local_composition": "fixed_local_composition",
                                 #"variable_local_composition": "variable_local_composition",
                                 "school_utilities": "school_utilities", "residential_utilities":"residential_utilities",
                                 "pos":"pos"})

        else:
            self.datacollector = DataCollector(
                model_reporters={"agent_count":
                                     lambda m: m.schedule.get_agent_count(), "seg_index": "seg_index",
                                 "residential_segregation": "residential_segregation", "res_seg_index": "res_seg_index",
                                 "fixed_res_seg_index": "fixed_res_seg_index",
                                 "happy": "happy", "percent_happy": "percent_happy",
                                 "total_moves": "total_moves", "compositions0": "compositions0",
                                 "compositions1": "compositions1",
                                 "comp0": "comp0", "comp1": "comp1", "comp2": "comp2", "comp3": "comp3",
                                 "comp4": "comp4", "comp5": "comp5", "comp6": "comp6",
                                 "comp7": "comp7", "compositions": "compositions",
                                 "collective_utility": "collective_utility"
                                 },
                agent_reporters={"local_composition": "local_composition", "type": lambda a: a.type,
                                 "id": lambda a: a.unique_id,
                                 # "fixed_local_composition": "fixed_local_composition",
                                 # "variable_local_composition": "variable_local_composition",
                                 "pos": "pos"})




        # Calculate local composition
        # set size
        for school in self.schools:
            #school.get_local_school_composition()
            #cap = round(np.random.normal(loc=cap_max * self.avg_school_size, scale=self.avg_school_size * 0.05))
            cap = self.avg_school_size * self.cap_max
            school.capacity = int(cap)
            print("cap",self.avg_school_size, cap)
            segregation_index(self)
        #


        print("height = %d; width = %d; density = %.2f; num_schools = %d; minority_pc =  %.2f; "
              "f0 =  %.2f; f1 =  %.2f; M0 =  %.2f; M1 =  %.2f;\
        alpha =  %.2f; temp =  %.2f; cap_max =  %.2f; move = %s"%(height,
         width, density, self.num_schools,minority_pc,f0,f1, M0,M1,alpha,
                                       temp, cap_max, move))
        #, move, symmetric_positions,bounded, radius, schelling ) )# Options  for the model




        self.total_considered = 0
        self.running = True
        self.datacollector.collect(self)





    def calculate_all_distances_to_schools(self):
        """
        1. Update the household.Dij distance to school matrix after agents have moved
        2. Add new students to the school neighbourhood
        This is only required at the end of every step of the model because
            1. neighbourhood students are used in the segregation metrics
            2. each agent only moves once


        TODO: better to just keep a matrix which maps positions to schools instead of households to schools
        :return: dist

        """

        for school in self.schools:
            school.neighbourhood_students = []

        Dij = distance.cdist(np.array(self.household_locations), np.array(self.school_locations), 'euclidean')

        for household_index, household in enumerate(self.households):
            Dj = Dij[household_index,:]
            household.update_distance_to_schools(Dj)

            # Calculate distances of the schools - define the school-neighbourhood and compare
            # closer_school = household.schools[np.argmin(household.)]
            closer_school_index = np.argmin(household.Dj)
            household.closer_school = self.schools[closer_school_index]
            household.closer_school.update_school_neighbourhood_students(household)

        return(Dij)




    def calculate_all_distances_to_neighbourhoods(self):
        """

        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist

        """
        for household_index, household in enumerate(self.households):

            # Calculate distances of the schools - define the school-neighbourhood and compare
            # closer_school = household.schools[np.argmin(household.)]
            household.closer_neighbourhood = self.get_closer_neighbourhood_from_position(household.pos)
            household.closer_neighbourhood.neighbourhood_students_indexes.append(household_index)


        # just sanity check
        # for i, neighbourhood in enumerate(self.neighbourhoods):
        #     students = neighbourhood.neighbourhood_students_indexes
        #     print("students,",i, len(students))





    def set_positions_to_school(self):
        '''
        calculate closer school from every position on the grid
        Euclidean or gis shortest road route
        :return: dist
        '''
        distance_dict = {}
        # Add the agent to a random grid cell

        all_grid_locations = []

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                all_grid_locations.append( (x,y) )


        Dij = distance.cdist(np.array(all_grid_locations), np.array(self.school_locations), 'euclidean')

        for i, pos in enumerate(all_grid_locations):
            Dj = Dij[i, :]
            (x,y) = pos
            # Calculate distances of the schools - define the school-neighbourhood and compare
            # closer_school = household.schools[np.argmin(household.)]
            closer_school_index = np.argmin(Dj)
            self.closer_school_from_position[x][y] = closer_school_index

        #print("closer_school_by_position",self.closer_school_from_position)

    def set_positions_to_neighbourhood(self):
        '''
        calculate closer neighbourhood centre from every position on the grid
        Euclidean or gis shortest road route
        :return: dist
        '''
        distance_dict = {}
        # Add the agent to a random grid cell

        all_grid_locations = []

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                all_grid_locations.append((x, y))

        Dij = distance.cdist(np.array(all_grid_locations), np.array(self.neighbourhood_locations), 'euclidean')

        for i, pos in enumerate(all_grid_locations):
            Dj = Dij[i, :]
            (x, y) = pos
            # Calculate distances of the schools - define the school-neighbourhood and compare
            # closer_school = household.schools[np.argmin(household.)]
            closer_neighbourhood_index = np.argmin(Dj)
            self.closer_neighbourhood_from_position[x][y] = closer_neighbourhood_index

        #print("closer_school_by_position", self.closer_school_from_position)



    def get_closer_school_from_position(self, pos):
        """
        :param pos: (x,y) position
        :return school: school object closest to this position
        """
        (x, y) = pos
        school_index = self.closer_school_from_position[x][y]
        school = self.get_school_from_index(school_index)

        return (school)



    def get_closer_neighbourhood_from_position(self, pos):
        """
        :param pos: (x,y) position
        :return school: school object closest to this position
        """
        (x, y) = pos
        neighbourhood_index = self.closer_neighbourhood_from_position[x][y]
        neighbourhood = self.get_neighbourhood_from_index(neighbourhood_index)

        return (neighbourhood)


    def get_school_from_index(self, school_index):
        """
        :param self: obtain the school object using the index
        :param school_index:
        :return: school object
        """

        return(self.schools[int(school_index)])

    def get_neighbourhood_from_index(self, neighbourhood_index):
        """
        :param self: obtain the school object using the index
        :param school_index:
        :return: school object
        """

        return (self.neighbourhoods[int(neighbourhood_index)])


    def get_households_from_index(self, household_indexes):

        """
        Retrieve household objects from their indexes
        :param household_indexes: list of indexes to retrieve household objects
        :return: households: household objects
        """
        households = []
        for household_index in household_indexes:
            households.append(self.households[household_index])
        return(households)


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

        satisfaction = 0
        res_satisfaction=0
        print("happy", self.happy)
        print("total_considered", self.total_considered)


        # Once residential steps are done calculate school distances

        if self.schedule.steps <= self.residential_steps or self.schedule.steps in [0,1]  :
            # during the residential steps keep recalculating the school neighbourhood compositions
            # this is required for the neighbourhoods metric

            #print("recalculating neighbourhoods")
            # TODO: check this, not sure if this and the recalculation below is needed
            for school in self.schools:
                school.neighbourhood_students = []
            for neighbourhood in self.neighbourhoods:
                neighbourhood.neighbourhood_students_indexes = []



            # update the household locations after a move
            self.household_locations = []
            for i, household in enumerate(self.households):
                self.household_locations.append(household.pos)


            self.calculate_all_distances_to_schools()
            self.calculate_all_distances_to_neighbourhoods()
            #print("all", self.calculate_all_distances()[i, :])


            self.residential_segregation = segregation_index(self, unit="neighbourhood")
            self.res_seg_index = segregation_index(self, unit="agents_neighbourhood")
            self.fixed_res_seg_index = segregation_index(self, unit="fixed_agents_neighbourhood", radius=1)
            self.school_neighbourhood_seg_index= segregation_index(self, unit="school_neighbourhood")

            res_satisfaction = np.mean(self.res_satisfaction)



        satisfaction =0
        # calculate these after residential_model
        if self.schedule.steps>self.residential_steps:
            self.collective_utility = calculate_collective_utility(self)
            print(self.collective_utility)
            self.seg_index = segregation_index(self)
            satisfaction = np.mean(self.satisfaction)

            self.calculate_all_distances_to_schools()



        else:
            self.collective_res_utility = calculate_res_collective_utility(self)




        print("seg_index", "%.2f"%(self.seg_index), "var_res_seg", "%.2f"%(self.res_seg_index), "neighbourhood",
              "%.2f"%(self.residential_segregation), "fixed_res_seg_index","%.2f"%(self.fixed_res_seg_index), \
              "res_satisfaction %.2f" %res_satisfaction,"satisfaction %.2f" %satisfaction,\
              "school_neighbourhood %.2f" %self.school_neighbourhood_seg_index,\
              "average_like_fixed %.2f"%self.average_like_fixed,"average_like_var %.2f"%self.average_like_variable  )


        if self.happy == self.schedule.get_agent_count() or self.schedule.steps>200:
            self.running = False





        compositions = []

        # remove this?
        for school in self.schools:
            self.my_collector.append([self.schedule.steps, school.unique_id, school.get_local_school_composition()])
            self.compositions = school.get_local_school_composition()
            compositions.append(school.get_local_school_composition()[0])
            compositions.append(school.get_local_school_composition()[1])

            self.compositions1 = int(school.get_local_school_composition()[1])
            self.compositions0 = int(school.get_local_school_composition()[0])
            #print("school_students",school.neighbourhood_students)

        #print("comps",compositions,np.sum(compositions) )
        [self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7] = compositions[0:8]
        # collect data
        #
        self.datacollector.collect(self)
        print("moves",self.total_moves, "res_moves", self.res_moves, "percent_happy", self.percent_happy)

        for i, household in enumerate(self.households):
            household.school_utilities = []
            household.residential_utilities = []








