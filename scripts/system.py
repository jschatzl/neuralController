class PT1:
    def __init__(self, K: float, T: float, dt: float):
        self.K = K
        self.T = T
        self.dt = dt
        self.y = 0.0  

    def step(self, u: float) -> float:
        self.y += (self.dt / self.T) * (self.K * u - self.y)
        return self.y
    
class PT2:
    def __init__(self, K: float, T: float, D: float, dt: float):
        self.K = K
        self.T = T
        self.D = D
        self.dt = dt
        self.y = 0.0
        self.dy = 0.0

    def step(self, u: float) -> float:
        ddy = (self.K * u - self.y - 2 * self.D * self.T * self.dy) / (self.T ** 2)
        self.dy += self.dt * ddy
        self.y  += self.dt * self.dy
        return self.y

class I:
    def __init__(self, K: float, T: float):
        self.K = K
        self.T = T
        self.yn = 0.0

    def step(self, u: float) -> float:
        return self.yn + (self.K * u * self.T)
