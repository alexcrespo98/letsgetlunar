# scripts/lunar_env.py
# lunar orbital insertion gymnasium environment
# alex crespo | 2026

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── PHYSICAL CONSTANTS ───────────────────────────────────────────────────────
R_MOON = 1734.7e3
G0     = 1.62
THRUST = 90000.0
MASS   = 300.0
TM     = THRUST / MASS
ALT_T  = 400e3
D_T    = R_MOON + ALT_T
MU     = G0 * R_MOON**2
VC     = np.sqrt(MU / D_T)

# ── INITIAL CONDITIONS ───────────────────────────────────────────────────────
X1_0   = R_MOON
X2_0   = 0.0
X3_0   = 0.0
X4_0   = 0.0
BETA_0 = np.radians(90.0)

# ── OBSERVATION GAINS ───────────────────────────────────────────────────────
GAIN_ALT = 0.0022222
GAIN_VR  = 4e-5
GAIN_VT  = 4e-5

# ── RATE LIMITER ─────────────────────────────────────────────────────────────
TS        = 0.05
MAX_DBETA = np.radians(2000.0 * TS)

# ── EPISODE PARAMETERS ───────────────────────────────────────────────────────
TF_MAX    = 160.0
MAX_STEPS = int(TF_MAX / TS)

# ── SUCCESS TOLERANCES ──────  ────────────────────────────────────────────────
TOL_ALT = 20e3
TOL_VR  = 50.0
TOL_VT  = 100.0


class LunarOrbitEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, reward_fn='shaped', exploring_starts=False,
                 reward_weights=(0.4, 0.3, 0.3), gaussian_widths=(50.0, 200.0, 200.0)):
        super().__init__()
        assert reward_fn in ('sparse', 'shaped', 'multiobjective')
        self.reward_fn        = reward_fn
        self.exploring_starts = exploring_starts
        self.reward_weights   = tuple(reward_weights)    # (w_alt, w_vr, w_vtan)
        self.gaussian_widths  = tuple(gaussian_widths)   # (sig_alt_km, sig_vr, sig_vtan)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-np.pi/2], dtype=np.float32),
            high=np.array([ np.pi/2], dtype=np.float32),
            dtype=np.float32
        )

        self.state      = None
        self.beta       = None
        self.t          = None
        self.step_count = None
        self.prev_beta  = None
        self.delta_beta = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.exploring_starts:
            self.state = np.array([
                R_MOON + self.np_random.uniform(0, 50e3),
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-500, 500),
                self.np_random.uniform(0.7 * VC, 1.3 * VC)
            ])
            self.beta = self.np_random.uniform(-np.pi/4, np.pi/2)
        else:
            self.state = np.array([X1_0, X2_0, X3_0, X4_0])
            self.beta  = BETA_0

        self.prev_beta  = self.beta
        self.delta_beta = 0.0
        self.t          = 0.0
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        beta_cmd = float(np.clip(action[0], -np.pi/2, np.pi/2))

        self.prev_beta  = self.beta
        delta           = np.clip(beta_cmd - self.beta, -MAX_DBETA, MAX_DBETA)
        self.beta       = self.beta + delta
        self.delta_beta = delta

        self.state       = self._rk4_step(self.state, self.beta, TS)
        self.t          += TS
        self.step_count += 1

        obs     = self._get_obs()
        done    = self._is_done()
        success = self._is_success()
        reward  = self._compute_reward(done, success)
        info    = {
            'altitude_km': (self.state[0] - R_MOON) / 1e3,
            'vr':          self.state[2],
            'vtan':        self.state[3],
            'beta_deg':    np.degrees(self.beta),
            'success':     success,
            't':           self.t
        }
        return obs, reward, done, False, info

    def _eom(self, state, beta_rad):
        x1, x2, x3, x4 = state
        return np.array([
            x3,
            x4 / x1,
            x4**2 / x1 - MU / x1**2 + TM * np.sin(beta_rad),
            -x3 * x4 / x1           + TM * np.cos(beta_rad)
        ])

    def _rk4_step(self, state, beta_rad, dt):
        k1 = self._eom(state,             beta_rad)
        k2 = self._eom(state + 0.5*dt*k1, beta_rad)
        k3 = self._eom(state + 0.5*dt*k2, beta_rad)
        k4 = self._eom(state + dt*k3,     beta_rad)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _get_obs(self):
        alt_err_km = (self.state[0] - D_T) / 1e3
        vr         = self.state[2]
        vtan       = self.state[3]
        return np.array([
            alt_err_km * GAIN_ALT,
            vr         * GAIN_VR,
            vtan       * GAIN_VT
        ], dtype=np.float32)

    def _is_success(self):
        return (
            abs(self.state[0] - D_T) < TOL_ALT and
            abs(self.state[2])        < TOL_VR  and
            abs(self.state[3] - VC)   < TOL_VT
        )

    def _is_crashed(self):
        return self.state[0] < R_MOON * 0.999

    def _is_escaped(self):
        return self.state[0] > R_MOON + 600e3

    def _is_done(self):
        return (
            self._is_success() or
            self._is_crashed() or
            self._is_escaped() or
            self.step_count >= MAX_STEPS
        )

    def _compute_reward(self, done, success):
        if self.reward_fn == 'sparse':
            return self._reward_sparse(done, success)
        elif self.reward_fn == 'shaped':
            return self._reward_shaped(done, success)
        elif self.reward_fn == 'multiobjective':
            return self._reward_multiobjective(done, success)

    def _reward_sparse(self, done, success):
        if not done:
            return 0.0
        if success:
            return +1000.0
        if self._is_crashed():
            return -500.0
        return -100.0

    def _reward_shaped(self, done, success):
        alt_err  = (self.state[0] - D_T) / 1e3
        vr_err   = self.state[2]
        vtan_err = self.state[3] - VC

        r_alt  = -0.5  * (alt_err  / 400.0)**2
        r_vr   = -0.3  * (vr_err   / 1000.0)**2
        r_vtan = -0.2  * (vtan_err / VC)**2

        dbeta_deg = np.degrees(abs(self.delta_beta))
        r_ctrl = -0.05 * (dbeta_deg / 90.0)**2

        reward = r_alt + r_vr + r_vtan + r_ctrl

        if success:
            reward += 500.0
        elif self._is_crashed():
            reward -= 200.0
        elif self._is_escaped():
            reward -= 100.0

        return float(reward)

    def _reward_multiobjective(self, done, success):
        alt_err_km = abs(self.state[0] - D_T) / 1e3
        vr_err     = abs(self.state[2])
        vtan_err   = abs(self.state[3] - VC)
        dbeta_deg  = np.degrees(abs(self.delta_beta))

        w_alt, w_vr, w_vtan           = self.reward_weights
        sig_alt, sig_vr, sig_vtan     = self.gaussian_widths

        g_alt  = np.exp(-0.5 * (alt_err_km / sig_alt )**2)
        g_vr   = np.exp(-0.5 * (vr_err     / sig_vr  )**2)
        g_vtan = np.exp(-0.5 * (vtan_err   / sig_vtan)**2)

        r_cont = -0.10 * (dbeta_deg / 90.0)**2
        r_time = -0.001

        reward = w_alt*g_alt + w_vr*g_vr + w_vtan*g_vtan + r_cont + r_time

        if success:
            reward += 1000.0
        elif self._is_crashed():
            reward -= 500.0
        elif self._is_escaped():
            reward -= 200.0

        return float(reward)