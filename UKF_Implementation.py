import numpy as np
import scipy.linalg as la
from copy import deepcopy as dc
import threading
from threading import Lock


class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        """
        Initializes the unscented kalman filter
        :param num_states: int, the size of the state
        :param process_noise: the process noise covariance per unit time, should be num_states x num_states
        :param initial_state: initial values for the states, should be num_states x 1
        :param initial_covar: initial covariance matrix, should be num_states x num_states
        :param alpha: UKF tuning parameter, determines spread of sigma points
        :param k: UKF tuning parameter
        :param beta: UKF tuning parameter
        :param iterate_function: function that predicts the next state
        """
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function

        # ------------------------------------------------------------------
        # TODO: Calculate self.lambd
        # Formula: alpha^2 * (n_dim + k) - n_dim
        # ------------------------------------------------------------------
        self.lambd = (alpha**2)*(self.n_dim + k) - self.n_dim # Replace with formula

        self.covar_weights = np.zeros(self.n_sig)

        self.mean_weights = np.zeros(self.n_sig)

        # ------------------------------------------------------------------
        # TODO: Initialize Weights
        # 1. Set self.covar_weights[0] and self.mean_weights[0]
        #    (Remember covar_weights[0] includes the beta term)
        # 2. Use a loop to set weights for the rest of the sigma points (1 to n_sig)
        # ------------------------------------------------------------------
        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - alpha**2 + beta) # Covariance weight for mean

        self.mean_weights[0] = self.lambd / (self.n_dim + self.lambd) # Mean weight for mean

        for i in range (1, 2*self.n_dim+1):
            self.mean_weights[i] = 1/(2*(self.lambd + self.n_dim)) # Mean weight for ith sigma point
            self.covar_weights[i] = 1/(2*(self.lambd + self.n_dim)) # Covariance weight for ith sigma point
        # ------------------------------------------------------------------
        # TODO: Generate Initial Sigmas
        # Call the __get_sigmas helper function and store in self.sigmas
        # ------------------------------------------------------------------
        self.sigmas = __get_sigmas() # Replace with call

        self.lock = Lock()

    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_sig, self.n_dim))

        # ------------------------------------------------------------------
        # TODO: Generate Sigma Points
        # 1. Calculate the square root matrix of: (n_dim + lambd) * self.p
        #    Hint: Use scipy.linalg.sqrtm
        # 2. Set the first point (ret[0]) to self.x
        # 3. Loop through n_dim to set the remaining points:
        #    - Positive direction: self.x + sqrt_matrix_column
        #    - Negative direction: self.x - sqrt_matrix_column
        # ------------------------------------------------------------------
        sqrt_matrix = la.sqrtm((self.n_dim + self.lambd) * self.p).T 
        # Calculated square root of (L+λ)P and took its transpose as we need ith column
        # in the square root matrix so to ease indexing took transpose

        ret[0] = self.x

        for i in range (1, self.n_dim + 1):
            ret[i] = self.x + sqrt_matrix[i-1] # ith sigma point
            ret[i+self.n_dim] = self.x - sqrt_matrix[i-1] # (L+i)th sigma point
        
        return ret.T

    def update(self, states, data, r_matrix):
        """
        performs a measurement update
        :param states: list of indices (zero-indexed) of which states were measured
        :param data: list of the data corresponding to the values in states
        :param r_matrix: error matrix for the data
        """

        self.lock.acquire()

        num_states = len(states)

        # ------------------------------------------------------------------
        # TODO: Create Measurement Sigmas (y) and Mean (y_mean)
        # 1. Split self.sigmas to isolate the states being measured.
        # 2. Create 'y' (sigmas of the measured states).
        # 3. Create 'y_mean' (mean of the measured states).
        # ------------------------------------------------------------------
        y = self.sigmas[states, :] #Taking the columns of sigmas which are related to the measurement

        y_mean = np.zeros((len(states), 1))
        for i in range(self.n_sig):
            y_mean += self.mean_weights[i] * y[:, i:i+1]
        # Running a loop to add the weighted sum of yi's
        # ------------------------------------------------------------------
        # TODO: Calculate Differences
        # 1. Calculate y_diff: difference between y and y_mean
        # 2. Calculate x_diff: difference between self.sigmas and self.x
        # ------------------------------------------------------------------
        y_diff = np.zeros((num_states, self.n_sig))
        for i in range(self.n_sig):
            y_diff[:, i:i+1] = y[:, i:i+1] - y_mean

        x_diff = np.zeros((self.n_dim, self.n_sig))
        for i in range(self.n_sig):
            x_diff[:, i:i+1] = self.sigmas[:, i:i+1] - self.x


        # ------------------------------------------------------------------
        # TODO: Calculate Measurement Covariance (p_yy)
        # 1. Initialize p_yy (num_states x num_states).
        # 2. Sum the weighted outer products of y_diff.
        # 3. CRITICAL: Add the measurement noise (r_matrix) to p_yy.
        # ------------------------------------------------------------------
        p_yy = np.zeros((num_states, num_states))

        for i in range (self.n_sig):
            p_yy += self.covar_weights[i]*(y_diff[:, i:i+1] @ y_diff[1, i:i+1].T)
        # Running a loop to add the weighted sums

        p_yy += r_matrix # Adding the error matrix

        # ------------------------------------------------------------------
        # TODO: Calculate Cross Covariance (p_xy)
        # 1. Initialize p_xy (n_dim x num_states).
        # 2. Sum the weighted products of x_diff and y_diff.
        # ------------------------------------------------------------------
        p_xy = np.zeros((self.n_dim, num_states))

        for i in range (self.n_sig):
            p_xy += self.covar_weights[i]*(x_diff[:, i:i+1] @ y_diff[:, i:i+1].T)

        # ------------------------------------------------------------------
        # TODO: Kalman Gain and Update
        # 1. Calculate K = p_xy * inv(p_yy)
        # 2. Update self.x using K and residual (data - y_mean)
        # 3. Update self.p using K and p_yy
        # 4. Recalculate self.sigmas using __get_sigmas()
        # ------------------------------------------------------------------
        K = p_xy * la.inv(p_yy)

        y_actual = np.array(data).reshape(-1, 1)

        self.x += K @ (y_actual - y_mean)
        self.p -= K @ p_yy @ K.T
        self.sigmas = __get_sigmas()

        self.lock.release()

    def predict(self, timestep, inputs=[]):
        """
        performs a prediction step
        :param timestep: float, amount of time since last prediction
        """

        self.lock.acquire()

        # ------------------------------------------------------------------
        # TODO: Propagate Sigma Points
        # 1. Pass each column of self.sigmas through self.iterate() function.
        #    (Pass timestep and inputs to iterate).
        # 2. Store result in sigmas_out.
        # ------------------------------------------------------------------
        sigma_transformed = [self.iterate_function(self.sigmas[:, i:i+1], timestep, inputs) for i in range (self.n_sig)] 
        # Making a list of the transformed sigma points 
        # after passing the original sigmas trough the transforming funtion

        sigmas_out = np.hstack(sigma_transformed)
        # Converting the above list into a 2d np array

        # ------------------------------------------------------------------
        # TODO: Calculate Predicted Mean (x_out)
        # Calculate the weighted sum of sigmas_out using self.mean_weights.
        # ------------------------------------------------------------------
        x_out = np.zeros((self.n_dim, 1)) # Initializing x_out with zeros

        for i in range (self.n_sig):
            x_out += self.mean_weights[i] * sigmas_out[:, i:i+1]
        # Evaluating the weighted sum using a loop

        # ------------------------------------------------------------------
        # TODO: Calculate Predicted Covariance (p_out)
        # 1. Loop through sigma points.
        # 2. Calculate diff = sigma_point - x_out.
        # 3. p_out += weight * (diff dot diff.T).
        # ------------------------------------------------------------------
        p_out = np.zeros((self.n_dim, self.n_dim)) # Initializing the new covariance matrix using zeros

        for i in range (self.n_sig):
            p_out += self.covar_weights[i] * ((sigmas_out[:, i:i+1] - x_out) @ (sigmas_out[:, i:i+1] - x_out).T)
        # Evaluating the weighted sum using a loop

        # ------------------------------------------------------------------
        # TODO: Add Process Noise
        # Add (timestep * self.q) to p_out.
        # ------------------------------------------------------------------
        p_out += self.q * timestep # Adding the process noise

        # ------------------------------------------------------------------
        # TODO: Update State
        # Set self.sigmas, self.x, and self.p to the new values.
        # ------------------------------------------------------------------
        self.x = x_out
        self.p = p_out
        self.sigmas = __get_sigmas()
        # Updting x, sigma points and conariance matrix

        self.lock.release()

    def get_state(self, index=-1):
        """
        returns the current state (n_dim x 1), or a particular state variable (float)
        :param index: optional, if provided, the index of the returned variable
        :return:
        """
        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        """
        :return: current state covariance (n_dim x n_dim)
        """
        return self.p

    def set_state(self, value, index=-1):
        """
        Overrides the filter by setting one variable of the state or the whole state
        :param value: the value to put into the state (1 x 1 or n_dim x 1)
        :param index: the index at which to override the state (-1 for whole state)
        """
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        """
        Restarts the UKF at the given state and covariance
        :param state: n_dim x 1
        :param covar: n_dim x n_dim
        """

        with self.lock:
            self.x = state
            self.p = covar