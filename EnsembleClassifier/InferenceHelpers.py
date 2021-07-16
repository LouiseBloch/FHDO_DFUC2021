import DataHelpers
import AugmentationHelpers
import torch
import numpy as np
import pandas as pd

def inference_Model(device,data_dir,augmentations,image_size,model,results_file,batch_size):
    validDataset=DataHelpers.prepare_Valid_Dataset(data_dir)
    DFUCDataset=DataHelpers.initialize_DFUC_Inference_Dataloader()
    transformValid=AugmentationHelpers.load_valid_augmentations(image_size,augmentations)
    datasetValid=DFUCDataset(validDataset,transform=transformValid)
    validation_loader = torch.utils.data.DataLoader(dataset=datasetValid, shuffle=False, batch_size=batch_size)
    model.eval()
    model.to(device)
    predictions=None
    for i, data in enumerate(validation_loader, 0):
        inputs = data
        inputs=inputs.cuda()
        outputs = model(inputs)
        outputs=torch.nn.functional.softmax(outputs,dim=1)
        outputs=outputs.cpu().detach().numpy()
        if predictions is None:
            predictions=np.array(outputs)
        else:
            predictions=np.concatenate((predictions, outputs), axis=0)
    predictionsDF=pd.DataFrame(np.c_[validDataset["image"],predictions],columns=["image","none","infection","ischaemia","both"])
    predictionsDF=predictionsDF.sort_values("image")
    predictionsDF.to_csv(results_file,index=False)

