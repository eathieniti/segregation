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

from Agents import SchoolAgent, HouseholdAgent

print("mesa",mesa.__file__)




class SchoolModel(Model):
    '''
    Model class for the Schelling segregation model.
    '''

    def __init__(self, height=100, width=100, density=0.95, num_schools=64,minority_pc=0.5, homophily=3, f0=0.6,f1=0.6,\
                 M0=0.8,M1=0.8,T=0.75,
                 alpha=0.2, temp=0.1, cap_max=1.01, move="boltzmann", symmetric_positions=True,
                 residential_steps=0,schelling=False,bounded=False,
                 residential_moves_per_step=2000, school_moves_per_step = 2000,radius=3,proportional = False,
                 torus=False,fs=0.9, extended_data = False, school_pos=None, agents=None, sample=5 ):
        '''
        '''
        # Options  for the model
        self.height = height
        self.width = width
        print("h x w",height, width)
        self.density = density
        self.num_schools= num_schools
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
        self.school_pos = school_pos
        self.extended_data = extended_data
        self.sample = sample

        if fs!="eq":
            self.fs = fs
        else:
            self.fs = f0

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



        # if schools already supplied place them where they should be

        if self.school_pos:
            school_positions = self.school_pos
            self.school_locations = school_pos
            self.num_schools = len(school_pos)

        # otherwise calculate the positions
        else:
            if self.num_schools == 4:
                school_positions = [(width/4,height/4),(width*3/4,height/4),(width/4,height*3/4),(width*3/4,height*3/4)]
            elif self.num_schools == 9:
                n=6
                school_positions = [(width/n,height/n),(width*3/n,height*1/n),(width*5/n,height*1/n),(width/n,height*3/n),\
                                    (width*3/n,height*3/n),(width*5/n,height*3/n),(width*1/n,height*5/n),(width*3/n,height*5/n),\
                                    (width*5/n,height*5/n)]
            elif  self.num_schools == 16:
                school_positions = []
                n=8
                x1 = [1, 3, 5, 7]

                xloc = np.repeat(x1, 4)
                yloc = np.tile(x1, 4)

                for i in range(len(x1 * 4)):
                    school_positions.append((xloc[i] * height / n, yloc[i] * width / n))

            elif self.num_schools == 64:
                school_positions = []
                n=int(np.sqrt( self.num_schools)*2)
                print(n)
                x1 = range(1,int(n+1),2)

                xloc = np.repeat(x1, int(n/2))
                yloc = np.tile(x1, int(n/2))

                for i in range( self.num_schools):
                    school_positions.append((xloc[i] * height / n, yloc[i] * width / n))

            elif self.num_schools == 25:
                school_positions = []
                n=int(np.sqrt( self.num_schools)*2)
                print(n)
                x1 = range(1,int(n+1),2)

                xloc = np.repeat(x1, int(n/2))
                yloc = np.tile(x1, int(n/2))

                for i in range( self.num_schools):
                    school_positions.append((xloc[i] * height / n, yloc[i] * width / n))



        for i in range(self.num_schools):
            #Add the agent to a random grid cell


            if self.symmetric_positions or self.school_pos:
                pos = (int(school_positions[i][0]),int(school_positions[i][1]))
            else:
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                pos = (x,y)

            self.school_locations.append(pos)
            school = SchoolAgent(pos, self)
            self.grid.place_agent(school, school.unique_id)
            self.schools.append(school)
            self.schedule.add(school)

        # Set up households


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



        else:

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



                self.households.append(agent)
                self.schedule.add(agent)





        self.calculate_all_distances()

        for agent in self.households:
            random_school_index = random.randint(0, len(self.schools)-1)
            candidate_school = self.schools[random_school_index]
            agent.allocate(candidate_school,agent.Dj[random_school_index])



            #closer_school = self.schools[p.argmin(Dj)]
            #closer_school.students.append(agent)
           # agent.allocate(closer_school, np.min(Dj))
            #print(agent.school.unique_id)




        self.pi_jm = np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.local_compositions =  np.zeros(shape=(len(self.school_locations),len(self.household_types )))
        self.avg_school_size = round(density*width*height/(len(self.schools)))

        if self.extended_data:
            self.datacollector = DataCollector(
                model_reporters={"agent_count":
                                     lambda m: m.schedule.get_agent_count(), "seg_index": "seg_index",
                                 "residential_segregation": "residential_segregation", "res_seg_index":  "res_seg_index","fixed_res_seg_index":"fixed_res_seg_index",
                                 "happy": "happy", "percent_happy": "percent_happy",
                                 "total_moves": "total_moves", "compositions0": "compositions0",
                                 "compositions1": "compositions1",
                                         "comp0": "comp0", "comp1": "comp1", "comp2": "comp2", "comp3": "comp3", "comp4": "comp4", "comp5": "comp5", "comp6": "comp6",
                                 "comp7": "comp7","compositions": "compositions",
                                 "collective_utility":"collective_utility"
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
            school.capacity = cap
            print("cap",self.avg_school_size, cap)
            segregation_index(self)
        #

        print("height = %d; width = %d; density = %.2f; num_schools = %d; minority_pc =  %.2f; "
              "f0 =  %.2f; f1 =  %.2f; M0 =  %.2f; M1 =  %.2f;\
        alpha =  %.2f; temp =  %.2f; cap_max =  %.2f; move = %s; symmetric_positions = %s"%(height,
         width, density, num_schools,minority_pc,f0,f1, M0,M1,alpha,
                                       temp, cap_max, move, symmetric_positions ))

        self.total_considered = 0
        self.running = True
        self.datacollector.collect(self)




    def calculate_all_distances(self):
        '''
        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist
        '''

        Dij = distance.cdist(np.array(self.household_locations), np.array(self.school_locations), 'euclidean')

        for i, household in enumerate(self.households):
            Dj = Dij[i,:]
            household.Dj = Dj

            # Calculate distances of the schools - define the school-neighbourhood and compare
            # closer_school = household.schools[np.argmin(household.)]
            closer_school_index = np.argmin(household.Dj)
            household.closer_school = self.schools[closer_school_index]
            household.closer_school.neighbourhood_students.append(household)

        return(Dij)




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



        if self.schedule.steps <= self.residential_steps or self.schedule.steps ==1 :
            # during the residential steps keep recalculating the school neighbourhood compositions
            # this is required for the neighbourhoods metric

            #print("recalculating neighbourhoods")
            for school in self.schools:
                school.neighbourhood_students = []



            self.household_locations = []
            for i, household in enumerate(self.households):
                self.household_locations.append(household.pos)
                household.residential_utilities = []


            self.calculate_all_distances()
            #print("all", self.calculate_all_distances()[i, :])

            # for i, household in enumerate(self.households):
            #     print(household.calculate_distances())
            #     # Calculate distances of the schools - define the school-neighbourhood and compare
            #     # closer_school = household.schools[np.argmin(household.)]
            #     closer_school_index = np.argmin(household.Dj)
            #     household.closer_school = self.schools[closer_school_index]
            #     household.closer_school.neighbourhood_students.append(household)
            #
            #     # Initialize house allocation to school
            #     #household.move_school(closer_school_index, self.schools[closer_school_index])
            #


            self.residential_segregation = segregation_index(self, unit="neighbourhood")
            self.res_seg_index = segregation_index(self, unit="agents_neighbourhood")
            self.fixed_res_seg_index = segregation_index(self, unit="fixed_agents_neighbourhood", radius=1)
            res_satisfaction = np.mean(self.res_satisfaction)



        satisfaction =0
        # calculate these after residential_model
        if self.schedule.steps>self.residential_steps:
            self.collective_utility = calculate_collective_utility(self)
            print(self.collective_utility)
            self.seg_index = segregation_index(self)
            satisfaction = np.mean(self.satisfaction)

        for i, household in enumerate(self.households):
            household.school_utilities = []


        print("seg_index", "%.2f"%(self.seg_index), "var_res_seg", "%.2f"%(self.res_seg_index), "neighbourhood",
              "%.2f"%(self.residential_segregation), "fixed_res_seg_index","%.2f"%(self.fixed_res_seg_index), \
              "res_satisfaction %.2f" %res_satisfaction,"satisfaction %.2f" %satisfaction,\
              "average_like_fixed %.2f"%self.average_like_fixed,"average_like_var %.2f"%self.average_like_variable  )


        if self.happy == self.schedule.get_agent_count():
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

        #print("comps",compositions,np.sum(compositions) )
        [self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7] = compositions[0:8]
        # collect data
        #
        self.datacollector.collect(self)
        print("moves",self.total_moves, "res_moves", self.res_moves, "percent_happy", self.percent_happy)





