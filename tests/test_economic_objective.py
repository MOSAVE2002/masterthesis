from types import SimpleNamespace
import importlib
import unittest

import gurobipy as gp


base = importlib.import_module("03_Gurobi.build_fjsp")
economic = importlib.import_module("helper.economic_objective")


def _instance(due_date=20.0):
    return SimpleNamespace(
        num_machines=2,
        real_operations=[1],
        eligible_machines={1: [0, 1]},
        processing_times={(1, 0): 10.0, (1, 1): 5.0},
        predecessors={1: []},
        jobs={1: [1]},
        job_end_operations={1: 1},
        due_dates={1: due_date},
        machine_cost={0: 1.0, 1: 4.0},
        machine_speed={0: 1.0, 1: 1.0},
        weibull_alpha={0: 30.0, 1: 30.0},
        weibull_beta={0: 2.0, 1: 2.0},
        repair_rate={0: 0.5, 1: 0.5},
        repair_duration={0: 2.0, 1: 2.0},
        instance_generation_model="independent_machine_parameters_v1",
    )


class EconomicObjectiveTests(unittest.TestCase):
    def test_expected_repair_buffer_is_used_unscaled(self):
        model = gp.Model()
        model.Params.OutputFlag = 0
        completion = model.addVar(lb=0.0, name="completion")
        raw_buffer = model.addVar(lb=2.0, ub=2.0, name="raw_buffer")
        instance = SimpleNamespace(
            jobs={1: [1]},
            job_end_operations={1: 1},
            due_dates={1: 10.0},
        )
        variables = {"C": {1: completion}}
        economic.add_robust_due_date_constraints(
            model,
            variables,
            instance,
            {1: raw_buffer},
        )
        model.setObjective(completion, gp.GRB.MAXIMIZE)
        model.optimize()
        self.assertEqual(model.Status, gp.GRB.OPTIMAL)
        self.assertAlmostEqual(completion.X, 8.0)
        self.assertAlmostEqual(
            variables["job_expected_repair_buffers"][1].X, 2.0
        )
        self.assertNotIn("job_scaled_repair_buffers", variables)
        self.assertNotIn("repair_buffer_scale", variables)
        model.dispose()

    def test_processing_plus_operating_cost_and_due_date(self):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model, variables = base.build_fjsp(
            model, _instance(), facility_cost_per_time=1.0
        )
        model.optimize()
        self.assertEqual(model.Status, gp.GRB.OPTIMAL)
        self.assertGreater(variables["Y"][1, 0].X, 0.5)
        self.assertAlmostEqual(variables["processing_cost"].getValue(), 10.0)
        self.assertAlmostEqual(variables["operating_cost"].getValue(), 10.0)
        self.assertAlmostEqual(model.ObjVal, 20.0)
        self.assertLessEqual(variables["C"][1].X, 20.0)
        model.dispose()

    def test_hard_due_date_can_make_base_model_infeasible(self):
        model = gp.Model()
        model.Params.OutputFlag = 0
        model, _variables = base.build_fjsp(model, _instance(due_date=4.0))
        model.optimize()
        self.assertEqual(model.Status, gp.GRB.INFEASIBLE)
        model.dispose()


if __name__ == "__main__":
    unittest.main()
