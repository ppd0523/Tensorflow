import math

# Moving SD(Standard de..)
class MovSD:
    def __init__(self, window):
        self.list = [0] * window
        self.window = window
        self.avg = 0
        self.oldavg = 0
        self.var = 0
        self.oldvar = 0
        self.sd = 0
        self.oldsd = 0

    def update(self, new):
        self.oldavg = self.avg
        self.avg = (new - self.list[0])/self.window + self.oldavg
        self.oldvar = self.var
        self.var = (new- self.list[0])*(new-self.avg+self.list[0]-self.oldavg)/(self.window-1) + self.oldvar
        self.oldsd = self.sd
        self.sd = math.sqrt(self.var)
        self.list = self.list[1:] + [new]

class MovRMS:
    def __init__(self, window):
        self.list = [0] * window
        self.window = window
        self.rms2 = 0
        self.oldrms2 = 0
        self.rms = 0

    def update(self, new):
        self.oldrms2 = self.rms
        self.rms2 = (new - self.list[0])*(new + self.list[0])/self.window + self.oldrms2
        self.rms = math.sqrt(self.rms2)
        self.list = self.list[1:] + [new]
