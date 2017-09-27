from HiddenMarkovModel import HiddenMarkovModel

class Classifier:
    
    def __init__ (self, allModels):
        self.allModels = allModels
        
    def classify(self, points):
        classRes = None
        maxRes = None
        for model in self.allModels:
            ev = model.evaluate(points)
            if classRes is None or ev > maxRes:
                classRes = model
                maxRes = ev
        if maxRes == 0:
            return "This sequence doesn't have a model."
        return "Is it a sequence of " + classRes.name +"?"
        