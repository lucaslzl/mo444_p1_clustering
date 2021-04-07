from model import Model


class KMeans(Model):


    def __init__(self, distance=0.5):

        self.distance = distance


    def fit(self, x):
        raise Exception('Required method was not implemented.')


    def predict(self, x):
        raise Exception('Required method was not implemented.')

