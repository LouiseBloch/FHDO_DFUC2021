import torch
import timm

def load_model(model_name,pretrained,num_classes,drop_rate,device,multi_GPU):
    model = timm.models.create_model(model_name,pretrained=pretrained,num_classes=num_classes,drop_rate=drop_rate)
    for i in model.conv_stem.parameters():
        i.requires_grad=False
    for i in model.bn1.parameters():
        i.requires_grad=False
    for i in model.act1.parameters():
        i.requires_grad=False
    for i in model.blocks.parameters():
        i.requires_grad=False
    if multi_GPU==True:
        model= torch.nn.DataParallel(model)
    model.to(device)
    return(model)

def load_optimizer(optimizer_type,model,learning_rate):
    if optimizer_type=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type=="RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    return(optimizer)

def load_scaler():
    scaler = torch.cuda.amp.GradScaler()
    return(scaler)

def load_loss_function():
    criterion = torch.nn.CrossEntropyLoss()
    return(criterion)

def end_warm_up_phase(model,device):
    for i in model.parameters():
        i.requires_grad=True
        model.to(device)
    model.to(device)
    return(model)

def load_inference_model(model_name,pretrained,num_classes,drop_rate,device,model_file,multi_GPU):
    model = timm.models.create_model(model_name,pretrained=pretrained,num_classes=num_classes,drop_rate=drop_rate)
    if multi_GPU==True:
        model= torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_file))
    return(model)