import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from tqdm import tqdm

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000
            ):
    

    logging.info("start training")
    counter = 0

    for epoch in range(epoch_start, max_epochs):
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_bodypart, loss_action, loss_multiple, loss_tryplay, loss_touchball, loss_goalpos, loss_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_bodypart, loss_action, loss_multiple, loss_tryplay, loss_touchball, loss_goalpos, loss_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid"
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)


        ###################### TEST ###################
        prediction_file, loss_bodypart, loss_action, loss_multiple, loss_tryplay, loss_touchball, loss_goalpos, loss_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
            torch.save(state, path_aux)
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    
    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "features_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name
    actions = {}

    if True:
        for targets in dataloader:
            # Unpack targets
            _, _, _, _, _, _, _, mvclips, action = targets

            # Move to GPU
            mvclips = mvclips.cuda().float()
            
            if pbar is not None:
                pbar.update()

            # Extract features
            features = model(mvclips)

            # Save features
            if len(action) == 1:
                values = features[0].tolist()  # Convert tensor to list
                actions[action[0]] = values
            else:
                for i in range(len(action)):
                    values = features[i].tolist()  # Convert tensor to list
                    actions[action[i]] = values

        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  

    return os.path.join(model_name, prediction_file)

def evaluation(dataloader,
          model,
          set_name="test",
        ):
    
    model.eval()

    prediction_file = "features_" + set_name + ".json"
    data = {}
    data["Set"] = set_name
    actions = {}
           
    if True:
        for _, _, mvclips, action in dataloader:
            mvclips = mvclips.cuda().float()
            
            # Extract features
            features = model(mvclips)

            if len(action) == 1:
                values = features[0].tolist()  # Convert tensor to list
                actions[action[0]] = values
            else:
                for i in range(len(action)):
                    values = features[i].tolist()  # Convert tensor to list
                    actions[action[i]] = values

        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file
