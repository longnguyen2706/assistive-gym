from cma import CMA
import numpy as np
class IK_CMAES:
    def __init__(self, x0, sigma, max_evals=1000):
        self.es = CMA(x0, sigma, max_evals=max_evals)

    def forward_ik(self, solution):
        # return forward ik result here
        pass

    def manipulability(self, solution):
        # return manipulability here
        pass

    def cost_function(self, solution, end_effector_position):
        # return cost function here
        actual_position = self.forward_kinematics(solution)
        return np.linalg.norm(end_effector_position - actual_position)  # Euclidean distance

    def fit(self):
        while not self.es.stop():
            solutions = self.es.ask()
            self.es.tell(solutions, [self.cost_function(x) for x in solutions])
            self.es.disp()



