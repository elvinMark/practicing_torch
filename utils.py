import torch
import torch.nn as nn

def train(model,train_dl,test_dl,optim,sched,crit,dev,epochs=50,project=None,experiment=None):
    if project and experiment:
        try:
            import wandb
            wandb.init(
                project= project,
                name=experiment
            )
            logger = wandb.log
        except:
            logger = print

    best_acc=torch.tensor(0.)
        
    for epoch in range(epochs):
        train_loss = 0.
        correct = 0.
        total = 0.
        for k,(x,y) in enumerate(train_dl):
            x = x.to(dev)
            y = y.to(dev)
                
            optim.zero_grad()
            o = model(x)
            l = crit(o,y)
            train_loss += l
            l.backward()
            optim.step()
            correct += torch.sum(torch.argmax(o,axis=1) == y)
            total += len(y)
        
        sched.step()
        train_acc = correct / total
        test_loss, test_acc = validate(model,test_dl,crit,dev)
        best_acc = max(best_acc,test_acc)
        
        logger({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "test_loss": test_loss
            })
    logger({
        "best_test_acc": best_acc
        })
    
def validate(model,test_dl,crit,dev):
    tot_loss = 0.
    correct = 0.
    total = 0.
    
    for x,y in test_dl:        
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o,y)
        tot_loss += l
        top1 = torch.argmax(o,axis=1)
        correct += torch.sum(top1 == y)
        total += len(y)

    return tot_loss, correct/total
