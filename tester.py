from PythonLinearNonlinearControl import configs 
from PythonLinearNonlinearControl import envs
from PythonLinearNonlinearControl import models
from PythonLinearNonlinearControl import planners
from PythonLinearNonlinearControl import controllers
from PythonLinearNonlinearControl import runners
import numpy as np

reference_xtraj = np.array([[7.94948032e+01, 7.97730485e+01, 7.98273453e+01, 7.96833141e+01, 7.93678521e+01, 7.92417644e+01, 7.93004896e+01, 7.94969711e+01],
                                [5.88653178e+03, 5.88382386e+03, 5.88126189e+03, 5.87886337e+03, 5.87666328e+03, 5.87460599e+03, 5.87270660e+03, 5.87097705e+03],
                                [2.46735892e+01, 2.62253133e+01, 2.46735892e+01, 2.31218650e+01, 2.15701409e+01, 2.00184168e+01, 1.84666926e+01, 1.69149685e+01],
                                [-1.54546825e+00, -1.46422017e+00, -1.54546825e+00, -1.62671634e+00, -1.70796443e+00, -1.62671634e+00, -1.54546825e+00, -1.46422017e+00]]).T

x0 = np.array([7.8696968e+01, 5.8860771e+03, 2.5599178e+01, -1.6295671e+00])

bad_plan = np.array([[15., -15., -15., -15., -15., -15., -15.],
                    [0.78539819, -0.78539819, -0.78539819, -0.78539819, 0.78539819, 0.78539819, 0.78539819]]).T

def run_tracker(reference_xtraj, x0, plot=True, bad_plan=None):
    u1_bds = [-20, 5]
    u2_bds = [-np.pi/4, np.pi/4]

    R = np.diag([0.01, 0.01])
    Q = np.diag([1e1, 1e1, 0, 0.01])
    Sf = np.diag([1e2, 1e2, 0, 0.01])

    config = configs.TwoWheeledConfigModule(R, Q, Sf)
    env = envs.DubinsTrackEnv(reference_xtraj[1:,:], x0, u1_bds=u1_bds, u2_bds=u2_bds)
    model = models.ExtDubinsModel(config)
    controller = controllers.iLQR(config, model)
    planner = planners.ClosestPointPlanner(config)
    runner = runners.ExpRunner()

    history_x, history_u, history_g = runner.run(env, controller, planner)
    if plot:
        assert bad_plan is not None, 'need the HJ plan to plot'
        env.plot_traj(bad_plan)
    
    return history_x, history_u

xtraj, utraj = run_tracker(reference_xtraj, x0, bad_plan=bad_plan)
print(len(utraj))
print(len(xtraj))