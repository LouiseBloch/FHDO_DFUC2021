import ModelHelpers
import torch
import time
import sklearn.metrics
import numpy as np
def iterate_epochs(train_loader,valid_loader,model_name,optimizer_type,drop_rate,num_training_samples,num_validation_samples,max_epochs,model_dir,device, learning_rate,learning_rate_warm_up,use_scheduler,num_warm_up_epochs,pretrained,NUM_CLASSES,multi_GPU):
    model=ModelHelpers.load_model(model_name,pretrained,NUM_CLASSES,drop_rate,device,multi_GPU)
    optimizer=ModelHelpers.load_optimizer(optimizer_type,model,learning_rate_warm_up)
    scaler=ModelHelpers.load_scaler()
    criterion=ModelHelpers.load_loss_function()
    if use_scheduler==True:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    else:
        scheduler=None
    for epoch in range(1,max_epochs+1):
            
        model_path=model_dir+"/Model_Epoch_"+str(epoch)+".pth"
        train_One_Epoch(train_loader,valid_loader,model,optimizer,scaler,criterion,model_path,num_training_samples,num_validation_samples,epoch)
        if num_warm_up_epochs==0:
            model=ModelHelpers.end_warm_up_phase(model,device)
            optimizer=ModelHelpers.load_optimizer(optimizer_type,model,learning_rate)
            if use_scheduler==True:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        elif epoch == num_warm_up_epochs:
            model=ModelHelpers.end_warm_up_phase(model,device)
            optimizer=ModelHelpers.load_optimizer(optimizer_type,model,learning_rate)
            if use_scheduler==True:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        if use_scheduler:
            if epoch>num_warm_up_epochs:
                scheduler.step()
    return(model)
        
def train_One_Epoch(train_loader,valid_loader,model,optimizer,scaler,criterion,model_path,num_training_samples,num_validation_samples,epoch):
    #initialize epoch starting time
    start = time.time()
    #iterate over training and validation 
    if valid_loader==None:
        phases=["train"]
    else:
        phases=['train', 'validation']
    for phase in phases:
    #initialize predictions and labels of entire dataset, to calculate the F1-scores for the training and validation dataset for each epoch 
        preds_epoch=[]
        label_epoch=[]
        #choose model mode and initialize data loaders dependently if the training or validation dataset should be used
        if phase == 'train':
            model.train()
            loader=train_loader
        else:
            model.eval()
            loader=valid_loader
        #initialize loss of current epoch
        running_loss = 0.0
        #iterate over all minibatches of the dataset
        for i, data in enumerate(loader, 0):
                # get the images and labels of one mini batch and convert to GPU readable format
            inputs, labels = data
            inputs=inputs.cuda()
            labels=labels.cuda()
            # mixed precision
            with torch.cuda.amp.autocast():
                #calculate output predictions of model and loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if phase == 'train':
                # zero the parameter gradients
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                #calculate and update parameter updates using the optimizer
                scaler.step(optimizer)
                scaler.update()
            # calculate and append prediction classes and labels to calculate F1-scores of each epoch
        _, preds = torch.max(outputs, 1)
        preds_epoch.append(preds.cpu().detach().numpy())
        label_epoch.append(labels.data.cpu().detach().numpy())
        #calculate running loss

        running_loss += loss.detach() * inputs.size(0)
        #flatten predictions and labels of epoch
        preds_epoch = [item for sublist in preds_epoch for item in sublist]
        label_epoch = [item for sublist in label_epoch for item in sublist]
        #calculate epoch loss for training dataset
        if phase == 'train':
            epoch_loss = running_loss / num_training_samples
            accuracy=sklearn.metrics.accuracy_score(label_epoch,preds_epoch)*100
            f1=sklearn.metrics.f1_score(label_epoch,preds_epoch,average="macro")*100
        prefix=''
        if phase=="validation":
            prefix='val_'
            #calculate epoch loss for validation dataset
            epoch_loss = running_loss / num_validation_samples
            #early stopping restrictions
            valaccuracy=sklearn.metrics.accuracy_score(label_epoch,preds_epoch)*100
    
    #save trained model
    torch.save(model.state_dict(),model_path)
    #get end time of epoch
    end = time.time()
    #calculate and print runtime per epoch
    print("Epoch "+str(epoch)+": Accuracy: "+str(np.round(accuracy,3))+ " F1-score: "+str(np.round(f1,3))+" Runtime: "+str(np.round((end-start),3))+" s")
#    print('{:5.3f}s'.format(end-start))