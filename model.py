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

print("mesa",mesa.__file__)


class SchoolAgent(Agent):
    '''

    '''
    def __init__(self, pos, model):

        super().__init__(pos, model)

        self.students = []
        self.type = 2
        self.position = pos



        # measures
        self.local_composition = [0,0]
        # TODO: households for now: change to students
        self.capacity = 0

    def step(self):
        pass

    def get_local_composition(self):

        local_composition = self.get_counts(self.students)

        self.local_composition = local_composition

        self.current_capacity = np.sum(self.local_composition)

        return(local_composition)


    def get_counts(self,students):

        local_composition = [0,0]
        d = [student.type for student in students]

        for agent_type in range(len(self.model.household_types)):
            local_composition[agent_type] = d.count(agent_type)

        return(local_composition)




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
        self.T = 0.75
        self.children = 1
        self.school = None
        self.dist_to_school = None
        self.local_composition = None
        self.position = pos





    def calculate_distances(self):
        '''
        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist
        '''
        Dj = np.zeros((len(self.model.school_locations),1))
        for i, loc in enumerate(self.model.school_locations):
            Dj[i] = np.linalg.norm(np.array(self.position)- np.array(loc))
        self.Dj = Dj
        #print("calculating distances", Dj)



    def step(self):
        self.model.total_considered += 1
        if self.model.schedule.steps < self.model.residential_steps:
            residential_move = True
        else:
            residential_move = False


        if residential_move:
            #if self.model.total_considered < 500:
                # move residential
                U_res = self.get_res_satisfaction()
               # print("U_res",U_res)
                if U_res < self.T:
                    self.model.grid.move_to_empty(self)
                else:
                    self.model.res_happy += 1

        else:
            if self.model.total_considered < 500:
            # school moves
                # satisfaction in current school
                U = self.get_school_satisfaction(self.school, self.dist_to_school)

                # If unhappy, compared to threshold move:
                if U < self.T:
                    #print('unhappy')
                    if self.model.deterministic == True:
                        self.evaluate_move(U)
                    else:
                        self.evaluate_move_boltzmann(U)
                else:
                    self.model.happy += 1
                    self.model.percent_happy = self.model.happy/self.model.total_considered




    def get_res_satisfaction(self):
        x=0
        p=0

        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, radius=1)
        for neighbour in neighbours:
            if isinstance(neighbour, HouseholdAgent):
                p += 1
                if neighbour.type == self.type:
                    x += 1


        P = self.ethnic_utility(x=x, p=p)
        self.res_satisfaction = P

        return(P)

    def allocate(self, school, dist):

        self.school = school
        school.students.append(self)
        self.dist_to_school = dist



    def evaluate_move_boltzmann(self, U):

        # consider each school at random
        # for candidate_school in random.sample(self.model.schools, len(self.model.schools)):
        random_order = np.random.permutation(len(self.model.schools))
        for school_index in random_order:
            #print("index",school_index)

            candidate_school = self.model.schools[school_index]

            if candidate_school.current_capacity <= (candidate_school.capacity + 10) and candidate_school!=self.school:
                U_candidate = self.get_school_satisfaction(candidate_school,dist=self.Dj[school_index])
                #print("U_cand,U",U_candidate,U)
                pr_move = self.prob_move(U,U_candidate)

                if pr_move >= random.random():
                    self.model.total_moves +=1
                    # remove the student from the school

                    self.move_school(school_index,candidate_school)
                    break



    def evaluate_move(self,U, proportional = False):
        # choose proportional or deterministic

        utilities = []
        for school_index, candidate_school in enumerate(self.model.schools):

            # check whether school is eligible to move to
             #if candidate_school.current_capacity <= (candidate_school.capacity + 10) and candidate_school!=self.school:
             if candidate_school.current_capacity <= (candidate_school.capacity + 10):
                 utilities.append(self.get_school_satisfaction(candidate_school,dist=self.Dj[school_index]))
             else:
                 utilities.append(0)
             #print(self.pos, candidate_school.pos, candidate_school.unique_id,self.Dj[school_index] )

        probabilities = utilities / np.sum(utilities)
        #print("utilities",utilities)
        #print(probabilities)
        if proportional:
            index_to_move = np.random.choice(len(probabilities), p=probabilities)
        else:
            index_to_move = np.argmax(probabilities)

        # only
        if self.model.schools[index_to_move] != self.school:
            self.move_school(index_to_move,self.model.schools[index_to_move])
            self.model.total_moves +=1


    def move_school(self, school_index, new_school):
        self.school.students.remove(self)

        # update metrics for school - could be replaced by +-1
        self.school.get_local_composition()

        # allocate elsewhere
        self.allocate(new_school, self.Dj[school_index])
        # now update the new school
        self.school.get_local_composition()


    def prob_move(self,U, U_candidate):

        deltaU = U_candidate-U

        return(1/(1+np.exp(-deltaU/self.model.temp)))


    #@property
    def get_school_satisfaction(self, school, dist):
        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # For the schools we add the distance satisfaction

        x = school.get_local_composition()[self.type]
        p = np.sum(school.get_local_composition())
        dist = float(dist)

        P = self.ethnic_utility(x,p)


        D = (self.model.max_dist - dist) / self.model.max_dist
        #print("D", D)
        U = P**(self.model.alpha) * D**(1-self.model.alpha)
        #print("P,D,U",P,D,U)

        return(U)


    def ethnic_utility(self, x, p):

        # x: local number of agents of own group in the school or neighbourhood
        # p: total number of agents in the school or neighbourhood
        # satisfaction
        fp = float(self.f * p)
        #print("fp,x",fp,x)


        # print(x,p)
        P = 0

        if self.type in [0, 1]:

            # assymetric satisfaction for minority
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

    def __init__(self, height=100, width=100, density=0.8, num_schools=4,minority_pc=0.5, homophily=3, f0=0.6,f1=0.6, M0=0.8,M1=0.8,
                 alpha=0.4, temp=0.3, cap_max=1.5, deterministic=True, symmetric_positions=True):
        '''
        '''
        # Options  for the model
        self.height = height
        self.width = width
        self.density = density
        self.num_schools= num_schools
        self.homophily = homophily
        self.f = [f0,f1]
        self.M = [M0,M1]
        self.residential_steps =0


        self.household_types = [0, 1]
        self.symmetric_positions = symmetric_positions


        # choice parameters
        self.alpha = alpha
        self.temp = temp

        self.households = []
        self.schools = []

        self.minority_pc = minority_pc


        self.num_households = int(width*height*density)
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(height, width, torus=False)
        self.total_moves = 0

        self.deterministic = deterministic

        self.school_locations = []
        self.household_locations = []
        self.Dij = []


        self.happy = 0
        self.res_happy = 0
        self.percent_happy = 0
        self.seg_index = 0
        self.res_seg_index = 0
        self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7 = 0,0,0,0,0,0,0,0
        self.satisfaction = []

        self.compositions = []



        self.my_collector = []
        self.max_dist = self.height*np.sqrt(2)



        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        # Set up schools


        school_positions = [(width/4,height/4),(width*3/4,height/4),(width/4,height*3/4),(width*3/4,height*3/4)]
        print("locations",school_positions)
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
            print("position",pos,school.pos)
            self.schools.append(school)
            self.schedule.add(school)
            print("school_pos",school.unique_id)

        # Set up households


        # create household locations but dont create agents yet


        while len(self.household_locations) < self.num_households:

            #Add the agent to a random grid cell
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            pos = (x,y)

            if (pos not in (self.school_locations) ) and (pos not in self.household_locations):
                self.household_locations.append(pos)





        Dij = self.calculate_distances()

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

            #closer_school = self.schools[np.argmin(Dj)]
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
                             "happy": "happy", "percent_happy": "percent_happy",
                             "total_moves": "total_moves", "compositions0": "compositions0",
                             "compositions1": "compositions1",
                                     "comp0": "comp0", "comp1": "comp1", "comp2": "comp2", "comp3": "comp3", "comp4": "comp4", "comp5": "comp5", "comp6": "comp6",
                             "comp7": "comp7","compositions": "compositions"},
            agent_reporters={"local_composition": "local_composition", "type": lambda a: a.type,
                             "id": lambda a: a.unique_id, })


        # Calculate local composition
        # set size
        for school in self.schools:
            school.get_local_composition()
            cap = round(np.random.normal(loc=cap_max * self.avg_school_size, scale=self.avg_school_size * 0.05))

            school.capacity = cap
        segregation_index(self)
        #

        print("height = %d; width = %d; density = %.2f; num_schools = %d; minority_pc =  %.2f; homophily = %d; "
              "f0 =  %.2f; f1 =  %.2f; M0 =  %.2f; M1 =  %.2f;\
        alpha =  %.2f; temp =  %.2f; cap_max =  %.2f; deterministic = %s; symmetric_positions = %s"%(height,
         width, density, num_schools,minority_pc,homophily,f0,f1, M0,M1,alpha,
                                       temp, cap_max, deterministic, symmetric_positions ))

        self.total_considered = 0
        self.running = True
        self.datacollector.collect(self)




    def calculate_distances(self):
        '''
        calculate distance between school and household
        Euclidean or gis shortest road route
        :return: dist
        '''

        Dij = distance.cdist(np.array(self.household_locations), np.array(self.school_locations), 'euclidean')
        return(Dij)




    def step(self):
        '''
        Run one step of the model. If All agents are happy, halt the model.
        '''
        self.happy = 0  # Reset counter of happy agents
        self.res_happy = 0
        self.total_moves = 0
        self.total_considered = 0


        self.schedule.step()

        print("happy", self.happy)
        print("total_considered", self.total_considered)

        self.seg_index = segregation_index(self)
        print("seg_index", self.seg_index, "res_seg_index", self.res_seg_index)

        # Once residential steps are done calculate school distances



        if self.schedule.steps < self.residential_steps - 1 or self.schedule.steps ==1 :

            for school in self.schools:
                school.neighbourhood_students = []

            household_locations = []
            for household in self.households:
                household.calculate_distances()




        if self.happy == self.schedule.get_agent_count():
            self.running = False
        compositions = []
        for school in self.schools:
            print(self.schedule.steps, school.unique_id,"final_composition",school.get_local_composition())
            self.my_collector.append([self.schedule.steps, school.unique_id, school.get_local_composition()])
            self.compositions = school.get_local_composition()
            print(self.local_compositions)
            compositions.append(school.get_local_composition()[0])
            compositions.append(school.get_local_composition()[1])

            self.compositions1 = int(school.get_local_composition()[1])
            self.compositions0 = int(school.get_local_composition()[0])

        print("comps",compositions,np.sum(compositions) )
        [self.comp0,self.comp1,self.comp2,self.comp3,self.comp4,self.comp5,self.comp6,self.comp7] = compositions[0:8]
        # collect data
        self.datacollector.collect(self)
        print("moves",self.total_moves, "percent_happy", self.percent_happy)







def segregation_index(model):
    # pi_jm: proportions in unit j that belongs to group m, shape: (j,m) (schools,groups)
    # pm: proportion in group m (m,1)



    # tj: total number in group j - dim: (1,j)


    pi_jm = np.zeros(shape=(len(model.school_locations), len(model.household_types)))
    local_compositions = np.zeros(shape=(len(model.school_locations), len(model.household_types)))


    for s_ind, school in enumerate(model.schools):
        local_composition = school.get_local_composition()

        local_compositions[s_ind][:] = local_composition
        pi_jm[s_ind][:] = local_composition / np.sum(local_composition)

        model.local_compositions = local_compositions
        model.pi_jm = pi_jm
    print(local_compositions, pi_jm)


    T=np.sum(local_compositions)

    tj = np.sum(local_compositions,axis=1, keepdims=True)
    #print("tj,",tj)

    pm = np.sum(pi_jm,axis=0)/np.sum(pi_jm, keepdims=True)
    #print("pm",pm)

    E = np.sum(pm*np.log(1/pm))

    #print("tj/TE",tj / (T * E))
    #print("pi_jm",pi_jm)
    #print("pm",pm)
    #print("pi_jm/pm",pi_jm/pm)

    seg_index = np.sum(tj / (T*E) * pi_jm * np.ma.log(pi_jm/pm), axis=None)

    print("seg_index",seg_index)

    return(seg_index)


def dissimilarity_index(model):

    for s_ind, school in enumerate(model.schools):
        local_composition = school.get_local_composition()
        model.pi_jm[s_ind][:] = local_composition

    pi_jm = model.pi_jm
    T=np.sum(pi_jm)
    tj = np.sum(pi_jm,axis=0, keepdims=True)
    pm = np.sum(pi_jm, axis=1, keepdims=True)

