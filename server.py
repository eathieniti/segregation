from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from  mesa.visualization.modules.BarChartVisualization import BarChartModule
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.ModularVisualization import VisualizationElement


import numpy as np
from mesa.visualization.TextVisualization import (
    TextData, TextGrid, TextVisualization
)

from model import SchoolModel
height = 100
width = 100
color_by_school  = False
add_schools = False
class SchellingTextVisualization(TextVisualization):
    '''
    ASCII visualization for schelling model
    '''

    def __init__(self, model):
        '''
        Create new Schelling ASCII visualization.
        '''
        self.model = model

        grid_viz = TextGrid(self.model.grid, self.ascii_agent)
        happy_viz = TextData(self.model, 'happy')

        self.elements = [grid_viz, happy_viz]



    @staticmethod
    def ascii_agent(a):
        '''
        Minority agents are X, Majority are O.
        '''
        if a.type == 0:
            return 'O'
        if a.type == 1:
            return 'X'



class HistogramModule(VisualizationElement):

    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
        canvas_width,
        canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        for agent in model.schedule.agents:
            hist = [10,20,30]
            return [int(x) for x in hist]




class HappyElement(TextElement):
    '''
    Display a text count of how many happy agents there are.
    '''

    def __init__(self):
        pass

    def render(self, model):
        return "Happy agents: " + str(model.happy)




def get_school_colors(model):

    schools_ids = [school.unique_id for school in model.schools]
    colors_list = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc51","#2ecc41","#2ecc21","#2ecc81"][0:len(schools_ids)]
    colors_list = ["LightGreen", "Yellow", "MediumOrchid", "LightBlue", "Green", "Blue", "Orchid", "LightYellow",
                   "DarkOrchid","LightGreen", "Yellow", "MediumOrchid", "LightBlue","LightGreen", "Yellow", "MediumOrchid", "LightBlue"]
    school_colors = dict(zip(schools_ids, colors_list))

    return(school_colors)



def schelling_draw(agent):
    '''
    Portrayal Method for canvas
    '''
    if agent is None:
        return
    portrayal = {"Shape": "rect", "w": 1.0,"h":1.0, "Filled": True, "Layer": 0}


    school_colors = get_school_colors(agent.model)


    if agent.type == 0:

        if agent.school:

            if color_by_school:
                portrayal["Color"] = school_colors[agent.school.unique_id]
                portrayal["stroke_color"] = school_colors[agent.school.unique_id]

            else:
                portrayal["Color"] = ["#000080", "#000080"]
                portrayal["stroke_color"] = ["#000080", "#000080"]

            portrayal["Shape"] = "rect"
        else: # decorator agents that are not assigned to school this only shows their type
            portrayal["Layer"] = 1
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = True
            portrayal["Color"] = ["#000080", "#000080"]
            portrayal["stroke_color"] = ["#000080", "#000080"]
            portrayal["r"]=0.7



    elif agent.type == 1:

        if agent.school:  #
            if color_by_school:
                portrayal["Color"] = school_colors[agent.school.unique_id]
                portrayal["stroke_color"] = school_colors[agent.school.unique_id]
            else:
                portrayal["Color"] = ["#FF0000", "#FF0000"]
                portrayal["stroke_color"] = ["#FF0000", "#FF0000"]

            portrayal["Shape"] = "rect"
        else:  # decorator agents that are not assigned to school this only shows their type
            portrayal["Layer"] = 1
            portrayal["Shape"] = "circle"
            portrayal["Filled"] = True
            portrayal["Color"] = ["#FF0000", "#FF0000"]
            portrayal["stroke_color"] =  ["#FF0000", "#FF0000"]
            portrayal["r"] = 0.7

    elif agent.type == 2:
        portrayal["Layer"] = 2
        print(agent.unique_id)
        portrayal["Shape"]= "circle"
        portrayal["r"] = 1
        if add_schools:
            portrayal["Color"] = school_colors[agent.unique_id]
            portrayal["stroke_color"] = ["#000000", "#000000"]
        else:
            portrayal["stroke_color"] = ["white", "white"]
            portrayal["Color"] = ["white", "white"]


    #print(agent.type,portrayal)
    return portrayal


happy_element = HappyElement()
canvas_element = CanvasGrid(schelling_draw, height,width, 500, 500)

compositions_chart = BarChartModule([{"Label": "comp0", "Color": "Black"}, {"Label": "comp1", "Color": "Blue"},
                              {"Label": "comp2", "Color": "Black"}, {"Label": "comp3", "Color": "Blue"},
                              {"Label": "comp4", "Color": "Black"}, {"Label": "comp5", "Color": "Blue"},
                              {"Label": "comp6", "Color": "Black"}, {"Label": "comp7", "Color": "Blue"}])
happy_chart = ChartModule([{"Label": "percent_happy", "Color": "Black"}])
seg_chart = ChartModule([{"Label": "seg_index", "Color": "Black"}],canvas_height=200, canvas_width=600, )
res_seg_chart = ChartModule([{"Label": "res_seg_index", "Color": "Black"}],canvas_height=200, canvas_width=600, )
neighbourhood_seg_chart = ChartModule([{"Label": "residential_segregation", "Color": "Black"}],canvas_height=200, canvas_width=600, )
res_satisfaction_chart = ChartModule([{"Label": "res_satisfaction", "Color": "Black"}],canvas_height=200, canvas_width=600, )
satisfaction_chart = ChartModule([{"Label": "satisfaction", "Color": "Black"}],canvas_height=200, canvas_width=600, )



histogram = HistogramModule(list(range(10)), 200, 500)


model_params = {
    "height": height,
    "width": width,
    "density": UserSettableParameter("slider", "Agent density", 0.90, 0.1, 1.0, 0.01),
    "minority_pc": UserSettableParameter("slider", "Fraction minority", 0.5, 0.00, 1.0, 0.05),
    "f0": UserSettableParameter("slider", "f0", 0.70, 0.1,0.9,0.05),
    "f1": UserSettableParameter("slider", "f1",0.70, 0.1, 0.9, 0.05),
    "M0": UserSettableParameter("slider", "M0", 0.8, 0.1, 1, 0.1),
    "M1": UserSettableParameter("slider", "M1", 0.8, 0.1, 1, 0.1),
    "cap_max": UserSettableParameter("slider", "max capacity", 1.01, 1.0, 5, 0.1),
    "alpha": UserSettableParameter("slider", "alpha", 0.2, 0.0, 1.0, 0.1),
    "temp": UserSettableParameter("slider", "temp", 0.1, 0.0, 0.9, 0.1)
}

server = ModularServer(SchoolModel,
                       [canvas_element, compositions_chart, seg_chart, happy_chart,
                        res_seg_chart,neighbourhood_seg_chart, res_satisfaction_chart, satisfaction_chart],
                       "SchoolModel", model_params)
server.launch()
