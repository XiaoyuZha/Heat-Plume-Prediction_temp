from data.dataset_loading import init_data
from solver import Solver
from networks.unet_leiterrl import TurbNetG, UNet
from networks.dummy_network import DummyNet
from visualization.visualize_data import plot_sample
from torch.nn import MSELoss
import datetime as dt
import sys
import logging
import numpy as np

def run_experiment(n_epochs:int=1000, lr:float=5e-4, inputs:str="pk", model_choice="unet", name_folder_destination:str="dummy", dataset_name:str="small_dataset_test", 
    path_to_datasets = "/home/pelzerja/pelzerja/test_nn/datasets", overfit=True):
    
    time_begin = dt.datetime.now()
    
    # parameters of model and training
    loss_fn = MSELoss()
    n_epochs = n_epochs
    lr=lr
    reduce_to_2D=True
    reduce_to_2D_xy=True
    overfit=overfit

    # init data
    datasets_2D, dataloaders_2D = init_data(dataset_name=dataset_name, path_to_datasets=path_to_datasets,
        reduce_to_2D=reduce_to_2D, reduce_to_2D_xy=reduce_to_2D_xy,
        inputs=inputs, labels="t", overfit=overfit)

    # model choice
    in_channels = len(inputs)+1
    if model_choice == "unet":
        model = UNet(in_channels=in_channels, out_channels=1).float()
    elif model_choice == "fc":
        size_domain_2D = datasets_2D["train"].dimensions_of_datapoint
        if reduce_to_2D:
            # TODO order here or in dummy_network(size) messed up
            size_domain_2D = size_domain_2D[1:]
        # transform to PowerOfTwo
        size_domain_2D = [2 ** int(np.log2(dimension)) for dimension in size_domain_2D]

        model = DummyNet(in_channels=in_channels, out_channels=1, size=size_domain_2D).float()
    elif model_choice == "turbnet":
        model = TurbNetG(channelExponent=4, in_channels=in_channels, out_channels=1).float()
    else:
        print("model choice not recognized")
        sys.exit()
    # model.to(device)

    # train model
    if overfit:
        solver = Solver(model, dataloaders_2D["train"], dataloaders_2D["train"], 
                    learning_rate=lr, loss_func=loss_fn)
    else:
        solver = Solver(model, dataloaders_2D["train"], dataloaders_2D["val"], 
                    learning_rate=lr, loss_func=loss_fn)
    solver.train(n_epochs=n_epochs, name_folder=name_folder_destination)

    # visualization
    if overfit:
        error, error_mean = plot_sample(model, dataloaders_2D["train"], name_folder_destination, plot_name="plot_learned_test_sample")
    else:
        error, error_mean = plot_sample(model, dataloaders_2D["test"], name_folder_destination, plot_name="plot_learned_test_sample", plot_one_bool=False)
    
    # save model - TODO : both options currently not working
    # save(model, str(name_folder)+str(dataset_name)+str(inputs)+".pt")
    # save_pickle({model_choice: model}, str(name_folder)+"_"+str(dataset_name)+"_"+str(inputs)+".p")

    # TODO overfit until not possible anymore (dummynet, then unet)
    # therefor: try to exclude connections from unet until it overfits properly (loss=0)
    # TODO go over input properties (delete some, some from other simulation with no hps?)
    # TODO : data augmentation, 

    time_end = dt.datetime.now()
    print(f"Time needed for experiment: {(time_end-time_begin).seconds//60} minutes {(time_end-time_begin).seconds%60} seconds")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    cla = sys.argv
    kwargs = {}

    kwargs["dataset_name"] = "small_dataset_test"
    kwargs["path_to_datasets"] = "/home/pelzerja/pelzerja/test_nn/dataset_generation/datasets"
    kwargs["n_epochs"] = 1

    if len(cla) >= 2:
        kwargs["n_epochs"] = int(cla[1])
        if len(cla) >= 3:
            kwargs["lr"] = float(cla[2])
            if len(cla) >= 4:
                kwargs["model_choice"] = cla[3]
                if len(cla) >= 5:
                    kwargs["inputs"] = cla[4]
                    if len(cla) >= 6:
                        kwargs["name_folder"] = cla[5]
                        if len(cla) >= 7:
                            kwargs["dataset_name"] = cla[6]

    run_experiment(**kwargs)

    # vary lr, vary input_Vars