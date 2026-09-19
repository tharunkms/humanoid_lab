import casadi as ca
from casadi import sin, cos, pi
import math
import numpy as np
from time import time
from typing import List, Dict
 
 
class MPCComponent:
    Q_x = 5
    Q_y = 5
    Q_theta = 0.5 # Increased slightly to encourage turning
    R_v = 0.1
    R_omega = 0.1
    v_max = 0.6
    v_min = -0.4 # Allow driving backward to escape tight corners
    omega_max = pi / 4
    omega_min = -omega_max
    # Obstacle constraints are soft: violating them costs W_slack_lin * s + W_slack_quad * s^2.
    # The linear term makes the penalty exact (behaves like a hard constraint whenever the
    # problem is feasible), the quadratic term keeps it well conditioned when it is not.
    W_slack_lin = 1e3
    W_slack_quad = 1e4

    def __init__(self, N=20, sim_time=20, step_horizon=0.1, rob_diameter=0.3):
        self.N = N
        self.sim_time = sim_time
        self.step_horizon = step_horizon
        self.rob_diameter = 1 # Legacy var (kept for compatibility)
        self.rob_circle_offset = 0.25 # Distance from center to front/rear circles
        self.rob_circle_diameter = 0.40 # Increased from 0.35 for safety margin
        self.MAX_OBS = 10
        self.obs = None
        self.obs_len = 0
        self.goal_tolerance = 1e-1 # step() returns zero controls once this close to the target
        self.last_solve_ok = True
        self.max_slack = 0.0
 
    def DM2Arr(self, dm):
        return np.array(dm.full())
 
    def init_symbolic_vars(self):
        # State Symbolic Variables
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()
 
        # Control Symbolic Variables
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.numel()
 
        # Matrix containing all states over all time steps + 1 (since it is initial + predictions)
        self.X = ca.SX.sym("X", self.n_states, self.N + 1)
 
        # Matrix containing all control actions predictions
        self.U = ca.SX.sym("U", self.n_controls, self.N)

        # Obstacle slack, one per predicted step k = 1..N (shared by all obstacles/circles)
        self.S = ca.SX.sym("S", self.N)

        # Sizes of the X / U blocks inside the flat optimization vector [X, U, S]
        self.n_X = self.n_states * (self.N + 1)
        self.n_U = self.n_controls * self.N
 
        # Parameter vector containing initial and target states + MAX_OBS (x, y, diameter)
        self.P = ca.SX.sym("P", self.n_states + self.n_states + self.MAX_OBS * 3)
 
        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)
 
        # controls weights matrix
        self.R = ca.diagcat(self.R_v, self.R_omega)
 
    def init_cost_fn_and_g_constraints(self):
        # Basic System Mapper Function
        rhs = ca.vertcat(
            self.v @ cos(self.theta), self.v @ sin(self.theta), self.omega
        )  # right hand side
        self.f = ca.Function("f", [self.states, self.controls], [rhs])
 
        # Loop for defining objectve function,
        self.cost_fn = 0
        self.g = (
            self.X[:, 0] - self.P[: self.n_states]
        )  # first constraint element
 
        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]
            target_state = self.P[self.n_states : self.n_states * 2]
            self.cost_fn += (st - target_state).T @ self.Q @ (
                st - target_state
            ) + con.T @ self.R @ con
            st_next = self.X[:, k + 1]
            st_next_euler = st + (self.step_horizon * self.f(st, con))
            self.g = ca.vertcat(self.g, st_next - st_next_euler)
            
        # Slack penalty for soft obstacle constraints
        self.cost_fn += self.W_slack_lin * ca.sum1(self.S) + self.W_slack_quad * ca.sumsqr(self.S)

        # Add dynamic parameterized obstacle constraints.
        # k = 0 is skipped: X[:, 0] is pinned to the measured pose, so constraining it
        # would make the whole problem infeasible whenever the robot is already too close.
        obs_start_idx = self.n_states * 2
        for j in range(self.MAX_OBS):
            obs_x = self.P[obs_start_idx + j * 3]
            obs_y = self.P[obs_start_idx + j * 3 + 1]
            obs_diam = self.P[obs_start_idx + j * 3 + 2]

            for k in range(1, self.N + 1):
                slack = self.S[k - 1]
                x = self.X[0, k]
                y = self.X[1, k]
                theta = self.X[2, k]
                
                # Front circle
                x_front = x + self.rob_circle_offset * ca.cos(theta)
                y_front = y + self.rob_circle_offset * ca.sin(theta)
                constraint_front = (self.rob_circle_diameter / 2 + obs_diam / 2) - ca.sqrt(
                    (x_front - obs_x)**2 + (y_front - obs_y)**2
                )
                
                # Middle circle
                constraint_middle = (self.rob_circle_diameter / 2 + obs_diam / 2) - ca.sqrt(
                    (x - obs_x)**2 + (y - obs_y)**2
                )
                
                # Rear circle
                x_rear = x - self.rob_circle_offset * ca.cos(theta)
                y_rear = y - self.rob_circle_offset * ca.sin(theta)
                constraint_rear = (self.rob_circle_diameter / 2 + obs_diam / 2) - ca.sqrt(
                    (x_rear - obs_x)**2 + (y_rear - obs_y)**2
                )
                
                self.g = ca.vertcat(
                    self.g, constraint_front - slack, constraint_middle - slack, constraint_rear - slack
                )
 
    def init_solver(self):
        # Preparing the NLP
        OPT_variables = ca.vertcat(
            self.X.reshape(
                (-1, 1)
            ),  # -1 as param means that casadi will automatically find the number of row/columns
            self.U.reshape((-1, 1)),
            self.S,
        )
 
        nlp_prob = {
            "f": self.cost_fn,
            "x": OPT_variables,
            "g": self.g,
            "p": self.P,
        }
 
        opts = {
            "ipopt": {
                "max_iter": 500, # Bounded so a bad solve can't stall the 10 Hz loop
                "mu_strategy": "adaptive", # Converges much faster when starting close to obstacles
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
            },
            "print_time": 0,
        }
 
        # Initialize solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)
 
    def init_constraint_args(self):
        # Initialze Optimization Variables Constraints Vector
        lbx = ca.DM.zeros((self.n_X + self.n_U + self.N), 1)
        ubx = ca.DM.zeros((self.n_X + self.n_U + self.N), 1)
 
        # States Bounds
        lbx[
            0 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # X lower bound
        lbx[
            1 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # Y lower bound
        lbx[
            2 : self.n_states * (self.N + 1) : self.n_states
        ] = -ca.inf  # theta lower bound
 
        ubx[
            0 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # X upper bound
        ubx[
            1 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # Y upper bound
        ubx[
            2 : self.n_states * (self.N + 1) : self.n_states
        ] = ca.inf  # theta upper bound
 
        # Controls Bounds
        lbx[
            self.n_X : self.n_X + self.n_U : self.n_controls
        ] = self.v_min  # V lower bound
        lbx[
            self.n_X + 1 : self.n_X + self.n_U : self.n_controls
        ] = self.omega_min  # Omega lower bound
 
        ubx[
            self.n_X : self.n_X + self.n_U : self.n_controls
        ] = self.v_max  # V upper bound
        ubx[
            self.n_X + 1 : self.n_X + self.n_U : self.n_controls
        ] = self.omega_max  # Omega upper bound

        # Slack Bounds (s >= 0)
        lbx[self.n_X + self.n_U :] = 0
        ubx[self.n_X + self.n_U :] = ca.inf
 
        self.args = {
            "lbg": ca.DM.zeros(
                (self.n_states * (self.N + 1), 1)
            ),  # state update constraints must equal 0
            "ubg": ca.DM.zeros(
                (self.n_states * (self.N + 1), 1)
            ),  # state update constraints must equal 0
            "lbx": lbx,
            "ubx": ubx,
        }
 
        # Dynamic Obstacle Constraints (must be <= 0)
        g_obs_len = self.N * self.MAX_OBS * 3 # 3 circles per obstacle, k = 1..N
        lbg_obs = ca.DM.zeros(g_obs_len, 1)
        lbg_obs[0:g_obs_len] = -ca.inf # Lower bound is -inf
        self.args["lbg"] = ca.vertcat(self.args["lbg"], lbg_obs)

        ubg_obs = ca.DM.zeros(g_obs_len, 1) # Upper bound is 0
        self.args["ubg"] = ca.vertcat(self.args["ubg"], ubg_obs)
 
    def _pack_obstacles(self, obstacles: List[Dict[str, float]] = None):
        # Format obstacles into a flat parameter array
        obs_params = []
        if obstacles is None:
            obstacles = []
        
        for i in range(self.MAX_OBS):
            if i < len(obstacles):
                obs_params.extend([obstacles[i]["x"], obstacles[i]["y"], obstacles[i]["diameter"]])
            else:
                # Dummy obstacle far away
                obs_params.extend([1000.0, 1000.0, 0.0])
                
        return ca.DM(obs_params)

    def _solve(self, obstacles: List[Dict[str, float]] = None):
        """Solve the NLP from state_current towards state_target.

        Returns the optimal control sequence (n_controls x N), or None if IPOPT did not
        converge. On failure the warm start is reset so the bad iterate doesn't leak into
        the next solve.
        """
        self.args["p"] = ca.vertcat(self.state_current, self.state_target, self._pack_obstacles(obstacles))
        self.args["x0"] = ca.vertcat(
            ca.reshape(self.X0, self.n_X, 1),
            ca.reshape(self.u0, self.n_U, 1),
            ca.DM.zeros(self.N, 1),
        )
        sol = self.solver(
            x0=self.args["x0"],
            lbx=self.args["lbx"],
            ubx=self.args["ubx"],
            lbg=self.args["lbg"],
            ubg=self.args["ubg"],
            p=self.args["p"],
        )

        self.last_solve_ok = bool(self.solver.stats()["success"])
        if not self.last_solve_ok:
            self.u0 = ca.DM.zeros((self.n_controls, self.N))
            self.X0 = ca.repmat(self.state_current, 1, self.N + 1)
            return None

        u = ca.reshape(
            sol["x"][self.n_X : self.n_X + self.n_U],
            self.n_controls,
            self.N,
        )
        self.X0 = ca.reshape(
            sol["x"][: self.n_X],
            self.n_states,
            self.N + 1,
        )
        # > 0 means the plan cuts into an obstacle margin (e.g. robot started too close)
        self.max_slack = float(ca.mmax(sol["x"][self.n_X + self.n_U :]))
        return u

    def _shift_warm_start(self, u):
        self.u0 = ca.horzcat(
            u[:, 1:], ca.reshape(u[:, -1], -1, 1)
        )  # Recycling Previous Controls Predicition
        self.X0 = ca.horzcat(
            self.X0[:, 1:], ca.reshape(self.X0[:, -1], -1, 1)
        )  # Recycling States Matrix

    def step(self, state_current: np.ndarray, state_target: np.ndarray, obstacles: List[Dict[str, float]] = None):
        """Returns the control sequence; all zeros (stop) if at target or the solve failed."""
        self.state_current = ca.DM(state_current)
        self.state_target = ca.DM(state_target)
 
        u = ca.DM.zeros(self.n_controls, self.N)
        if ca.norm_2(self.state_current - self.state_target) > self.goal_tolerance:
            u_sol = self._solve(obstacles)
            if u_sol is not None:
                u = u_sol
                self._shift_warm_start(u)
 
        else:
            self.mpc_completed = True
 
        return u
 
    def init_sim_params(self):
        self.cat_states = self.DM2Arr(self.X0)
        self.cat_controls = self.DM2Arr(self.u0[:, 0])
        self.times = np.array([[0]])
        self.t0 = 0
        self.t = ca.DM(self.t0)
        self.mpc_iter = 0
 
    def prepare_step(self, state_init: np.ndarray):
        # self.init_symbolic_vars()
        # self.init_cost_fn_and_g_constraints()
        # self.init_solver()
        # self.init_constraint_args()
 
        self.mpc_completed = False
 
        self.state_init = ca.DM(state_init)
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N + 1)
 
        return
 
    def step_with_sim_params(
        self, state_current: np.ndarray, state_target: np.ndarray, obstacles: List[Dict[str, float]] = None
    ):
        self.state_current = ca.DM(state_current)
        self.state_target = ca.DM(state_target)
 
        u = ca.DM.zeros(self.n_controls, self.N)
        if ca.norm_2(self.state_current - self.state_target) > self.goal_tolerance:
            t1 = time()
            u_sol = self._solve(obstacles)
            if u_sol is not None:
                u = u_sol
            self.cat_states = np.dstack((self.cat_states, self.DM2Arr(self.X0)))
            self.cat_controls = np.vstack(
                (self.cat_controls, self.DM2Arr(u[:, 0]))
            )
            self.t = np.vstack((self.t, self.t0))
            self.t0 += self.step_horizon
            if u_sol is not None:
                self._shift_warm_start(u)
            t2 = time()
            self.times = np.vstack((self.times, t2 - t1))
            self.mpc_iter += 1
 
        else:
            self.mpc_completed = True
 
        return u
 
    def get_simulation_params(self):
        return {
            "cat_states": self.cat_states,
            "cat_controls": self.cat_controls,
            "times": self.times,
            "step_horizon": self.step_horizon,
            "N": self.N,
            "p_arr": np.array(
                [
                    self.state_init[0],
                    self.state_init[1],
                    self.state_init[2],
                    self.state_target[0],
                    self.state_target[1],
                    self.state_target[2],
                ]
            ),
            "obs": self.obs,
            "rob_diam": self.rob_diameter,
        }
 
    def simulate_step_shift(self, u, state_init):
        f_value = self.f(state_init, u[:, 0])
        return ca.DM.full(state_init + (self.step_horizon * f_value))
 