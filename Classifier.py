from HiddenMarkovModel import HiddenMarkovModel

class Classifier:
    
    def __init__ (self, allModels):
        self.allModels = allModels
        accuracy = 0
        for model in self.allModels:
            for obs in model.testData:
                cName = self.classify(obs)
                if cName == model.name:
                    accuracy = accuracy + 1
            print ('Accuracy of ' + model.name + ' gesture model is ' + str((accuracy/30)*100) + '%')
            accuracy = 0
    def classify(self, points):
        classRes = None
        maxRes = None
        for model in self.allModels:
            ev = model.evaluate(points)
            if classRes is None or ev > maxRes:
                classRes = model
                maxRes = ev
        return classRes.name
        