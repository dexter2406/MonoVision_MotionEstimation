import numpy as np

def relVelEst(relVelPrev, DList, fps_vid):
    DList = [i for i in DList if abs(i) < 20]  # [1, 2, 5, -5, -3]
    listLen = len(DList)  # 5
    DListP = [i for i in DList if i > 0]
    DListN = [i for i in DList if i < 0]
    critP = len(DListP)  # increasing, 3
    critN = listLen - critP

    if critP > critN:
        DOutput = DListP
        relVel = np.mean(DOutput) * fps_vid
    elif critP > critN:
        DOutput = DListN
        relVel = np.mean(DOutput) * fps_vid
    else:
        relVel = relVelPrev

    print(relVel)
