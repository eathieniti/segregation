# segregation
This project uses the python package mesa to develop an agent based model to simulate the racial segregation in schools.
Briefly the agents are parents that make decisions on which school entity to choose based on a utility function. 
The utility function depends on the the distance to school and the racial composition of the school. 



model.py: Holds the mesa object for the School segregation model 

Agents.py: Holds the mesa agent objects

util.py: util functions used by the model.py

run_statistics_v2.py: Run the model for different parameter values and save the results in pandas dataframes

