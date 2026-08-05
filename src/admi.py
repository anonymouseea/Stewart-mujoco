import numpy as np

class Admittance:
    def __init__(self, M: float, D: float, K: float, dt: float) -> None:
        self.M = np.asarray(M, dtype=float).reshape(6)
        self.D = np.asarray(D, dtype=float).reshape(6)
        self.K = np.asarray(K, dtype=float).reshape(6)
        self.dt = float(dt)

        if np.any(self.M == 0.0):
            raise ValueError("导纳控制的质量 M 不能为 0，否则会除以 0。")
               
        self.x = np.zeros(6, dtype=float)
        self.xdot = np.zeros(6, dtype=float)
        self.xdesir = np.zeros(6, dtype=float)
        #期望力
        self.F_desire =np.zeros(6, dtype=float)
    

    def set_state(self, x: np.ndarray, xdesir: np.ndarray):
        self.x = np.asarray(x, dtype=float).copy()
        self.xdesir = np.asarray(xdesir, dtype=float).copy()

    def set_force(self, F_desire: np.ndarray):
        self.F_desire = np.asarray(F_desire, dtype=float).copy()

    def set_stated(self, xdesir: np.ndarray):

        self.xdesir = np.asarray(xdesir, dtype=float)

    def update_martix(self,matrix):   
        self.matrix = np.asarray(matrix, dtype=float).copy()

    def update(self, F_ext: np.ndarray):
        F_ext = np.asarray(F_ext, dtype=float)
        F_err =  F_ext - self.F_desire
        
        xddot = (F_err- self.D * self.xdot - self.K * (self.x-self.xdesir) )/ self.M
        
        self.xdot += xddot * self.dt
        self.x += self.xdot * self.dt
        
        return self.x.copy()

