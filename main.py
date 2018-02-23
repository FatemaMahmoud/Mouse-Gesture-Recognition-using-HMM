import pickle
from HiddenMarkovModel import HiddenMarkovModel
from HmmFrame import HmmFrame
from Classifier import Classifier

with open('gestures_observations_dict.pickle', 'rb') as f:
    obs = pickle.load(f)

allModels = list()   
for i in range(0,7):
    tempDic = {'X' : list(), 'Y': list()}
    for j in range(i*100,(i+1)*100):
        tempDic['X'].append(obs['X'][j])
        tempDic['Y'].append(obs['Y'][j])
        
    if tempDic['Y'][0] == 'true':
        allModels.append(HiddenMarkovModel(2, 'True', tempDic))
    elif tempDic['Y'][0] == 'rightdown':
        allModels.append(HiddenMarkovModel(2, 'Right Down', tempDic))
    elif tempDic['Y'][0] == 'BottomLeftCorner':
        allModels.append(HiddenMarkovModel(2, 'Down Right', tempDic))
    elif tempDic['Y'][0] == 'leg':
        allModels.append(HiddenMarkovModel(2, 'Right Up', tempDic))
    elif tempDic['Y'][0] == 'etneen':
        allModels.append(HiddenMarkovModel(2, 'Left Down', tempDic))
    elif tempDic['Y'][0] == 'downside':
        allModels.append(HiddenMarkovModel(3, 'Up Right Down', tempDic))
    else :
        allModels.append(HiddenMarkovModel(4, 'circle', tempDic))   
    tempDic.clear()

classifier = Classifier(allModels)
hf = HmmFrame()
hf.classifier(classifier)
hf.mainloop()
