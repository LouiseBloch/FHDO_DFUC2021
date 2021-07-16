import pandas as pd
import numpy as np
def inference_ensemble(model_files,results_file):
    data_Test=pd.read_csv(model_files[0])
    res=np.zeros_like(data_Test.filter(["none","infection","ischaemia","both"]).to_numpy())
    index=0
    for i in model_files:
        data=pd.read_csv(i)
        res=res+data.filter(["none","infection","ischaemia","both"]).to_numpy()
        index+=1
    res=res/index
    predictionsDF=pd.DataFrame(np.c_[data_Test["image"],res],columns=["image","none","infection","ischaemia","both"])
    predictionsDF.to_csv(results_file,index=False)
