import torch
from block_tanh import MDN, mdn_loss_fn
import wandb
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import polars as pl
import numpy as np
import datetime

v_max, s_max = 35.0, 100.0  # m/s, m


# split data into training and testing data
def split_data(dataset):
    X = dataset.select(['leader_speed', 'follower_speed'])
    y = dataset.select(['spacing'])
    # training data: 80% of data; testing data: 20% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # convert polars dataframe to numpy array
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    # convert numpy array to tensor
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)
    X_variable = Variable(X_train)
    y_variable = Variable(y_train, requires_grad=False)
    return X_variable, y_variable, X_test, y_test


# MDN model for specific data
def MDN_model(dataset, config):
    epoch, number_of_Gaussian, hidden_dim = config['epoch'], config['number_of_Gaussian'], config['hidden_dim']
    wandb.log({"data size": dataset.shape[0]})
    # 2D input, 1D output
    mdn_model = MDN(input_dim=2, n_hidden=hidden_dim, n_gaussians=number_of_Gaussian)
    optimizer = optim.Adam(mdn_model.parameters())
    X_variable, y_variable, X_test, y_test = split_data(dataset)
    for i in range(epoch):
        pi_variable, sigma_variable, mu_variable = mdn_model(X_variable)
        loss = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_variable)
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(mdn_model.parameters(), 1.0)
        optimizer.step()
        # log metrics to wandb
        wandb.log({"loss": loss.data})
        # print('epoch: ', i, 'loss: ', loss.data)
    return X_test, y_test, mdn_model


def validate_MDN(X_test, y_test, mdn_model):
    # compute the learned Mixtures of Gaussians
    pi_variable, sigma_variable, mu_variable = mdn_model(X_test)
    # compute overall test loss
    loss_test = mdn_loss_fn(pi_variable, sigma_variable, mu_variable, y_test)
    # print('loss_test', loss_test.data)
    # log metrics to wandb
    wandb.log({"loss_test": loss_test.data})
    # sort the X_test
    X_test = X_test.data.numpy()
    sort_ind = np.lexsort((X_test[:, 1], X_test[:, 0]))
    X_test, y_test = X_test[sort_ind], y_test.data.numpy()[sort_ind]
    # split the X_test into subsets of small range of X_test[:, 0] (leader_speed) and X_test[:, 1] (spacing)
    all_input, all_output = [], []
    leader_speed_test, follower_speed_test = X_test[:, 0], X_test[:, 1]
    num_vl, num_vs = int(v_max), int(v_max)
    for i in range(num_vl):
        for j in range(num_vs):
            subset = (leader_speed_test >= i) & (leader_speed_test < i + 1) & (follower_speed_test >= j) & (
                    follower_speed_test < j + 1)
            X, y = X_test[subset], y_test[subset]
            if len(y) > 10:
                all_input.append(np.mean(X, axis=0))
                all_output.append(np.mean(y))
    # compute the learned Mixtures of Gaussians for each subset
    all_input, all_output = np.array(all_input), np.array(all_output)
    all_input = torch.tensor(all_input, dtype=torch.float32)
    pi, sigma, mu = mdn_model(all_input)
    pi, sigma, mu = pi.data.numpy(), sigma.data.numpy(), mu.data.numpy()
    # compute the mean of MDN and the RMSE
    predicted_gap = np.sum(pi * mu, axis=1)
    mean_RMSE = np.sqrt(np.mean((predicted_gap - all_output) ** 2))
    # print('mean_RMSE', mean_RMSE)
    wandb.log({"mean_RMSE": mean_RMSE})
    # # plot the predicted mean vs the true mean and y=x line
    # plt.scatter(all_output, predicted_follower_speed, c='b')
    # plt.plot([np.min(all_output), np.max(all_output)], [np.min(all_output), np.max(all_output)], c='r')
    # plt.xlabel('True E$[V_i|V_{i-1}]$ (m/s)')
    # plt.ylabel('Predicted E$[V_i|V_{i-1}]$ (m/s)')
    # plt.show()
    return loss_test, mean_RMSE


def main(dataset, config):
    model_type = config['model_type']
    # start a new wandb run to track this script
    run = wandb.init(
        project='MDN_P(Si|Vi,Vi-1)_Waymo',
        name='Exp_' + model_type,
        config=config,
    )
    X_test, y_test, mdn_model = MDN_model(dataset, config)
    # save the model with date time
    name = str(model_type) + '_model_' + datetime.datetime.now().strftime("%H%M")
    torch.save(mdn_model.state_dict(), name + '.pth')
    # Save as artifact for version control.
    artifact = wandb.Artifact(name, type='model')
    artifact.add_file(name + '.pth')
    run.log_artifact(artifact)
    # validate the model
    _, _ = validate_MDN(X_test, y_test, mdn_model)
    # Mark the run as finished
    wandb.finish()


if __name__ == '__main__':
    # define the hyperparameters
    all_model_type = ['hv_av', 'hv_hv', 'av_hv']
    for model_type in all_model_type * 3:
        # different parameters for different model types
        if model_type == 'hv_av':
            epoch, hidden_dim, number_of_Gaussian = 20000, 64, 10
        elif model_type == 'av_hv':
            epoch, hidden_dim, number_of_Gaussian = 20000, 64, 10
        elif model_type == 'hv_hv':
            epoch, hidden_dim, number_of_Gaussian = 15000, 128, 100
        config_dict = {
            'epoch': epoch,
            "dataset": "Waymo",
            'number_of_Gaussian': number_of_Gaussian,
            'hidden_dim': hidden_dim,
            'model_type': model_type
        }
        print("Now processing: ", model_type)
        # read the data
        data_path = './data/waymo/' + model_type + '_data.parquet'
        data = pl.read_parquet(data_path)
        main(data, config=config_dict)
