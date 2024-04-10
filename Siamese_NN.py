#%% Importing the necessary libraries
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.utils import make_grid
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
#%% classes and functions
class TwoImageDataset(Dataset):
    """
    This class is a dataset class for the LFW dataset. It takes the path to the pairs.txt file and returns a dataset
    that can be used with the DataLoader class. The pairs.txt file contains the pairs of images that should be compared
    and the label that indicates if the images are of the same person or not. The dataset returns a tuple of two images
    and the label. The images are read using the PIL library and are resized to 105x105 pixels. The images are also
    converted to grayscale. The images are then converted to PyTorch tensors and normalized to the range [-1, 1].
    params: 
        pairs_text: str - The path to the pairs.txt file
        augmentations:defualt = False - a torchvision.transforms object that contains the augmentations that should be
        applied to the images. If no augmentations should be applied, the default value should be used.
    """
    
    def __init__(self, pairs_text,augmentations = False):
        self.pairs_text = pairs_text
        with open(self.pairs_text) as file:
            # Read the file and split the lines into a list of lists
          self.samples = [line.rstrip('\n').split('\t') for line in file][1:] 
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.read_images(self.samples[idx])

    def path(self, name, pic_num):
        # return the path to the image file based on the name and the picture number
        return f'lfw2/{name}/{name}_{pic_num.zfill(4)}.jpg' 
    
    def read_image(self, name, pic_num):
        # Open the image using PIL
        img = Image.open(self.path(name, pic_num)).convert('L')  # Convert('L') converts it to grayscale
        # Resize the image
        img = img.resize((105, 105)) 

        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range and to tensor
            transforms.Normalize((0.5), (0.5))  # Imgae Normalization with mean and std of 0.5
            ])
        img = transform(img) # Apply the transform to the image
        if self.augmentations is not False: # If augmentations are defined apply them
            assert not isinstance(type(self.augmentations),transforms.Compose), "The augmentations should be a torchvision.transforms.Compose object"
            img = self.augmentations(img)

        return name,img


    def read_images(self, sample):
      if len(sample) == 3: # If the sample is a positive sample (i.e. the images are of the same person)
        img1 = self.read_image(sample[0], sample[1])
        img2 = self.read_image(sample[0], sample[2])
        label = torch.tensor([1.])
        return (img1, img2, label)
      else:
        # If the sample is a negative sample (i.e. the images are of different people)
        assert len(sample) == 4, "The sample should contain 3 or 4 elements"
        img1 = self.read_image(sample[0], sample[1])
        img2 = self.read_image(sample[2], sample[3])
        label = torch.tensor([0.])
        return (img1, img2, label)
    
class SiameseNN(nn.Module):
    def __init__(self,arch):
        """
        This class is a Siamese Neural Network that takes two images as input and returns a single value that indicates
        if the images are of the same person or not. The architecture of the network can be defined using the arch
        parameter. The method parameter defines the weight initialization method that should be used. The class contains
        three main parts: the convolutional part, the fully connected part, and the output part.
        params:
            arch: str - The architecture of the network. The following architectures are supported:
                - "paper" - The architecture described in the paper
                - "arch_1" - A custom architecture
                - "arch_2" - A custom architecture
            method: str - The weight initialization method that should be used. The following methods are supported:
                - "paper" - The weight initialization method described in the paper
                - "xavier" - The Xavier weight initialization method
        """
        super(SiameseNN, self).__init__()
        
        assert arch  in ["paper", "arch_1", "arch_2"], "The arch parameter should be one of 'paper', 'arch_1', or 'arch_2'"
        if arch == "paper":
            self.arch = "paper"
            self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (10,10)),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64, 128, (7,7)),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128, 128, (4,4)),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128, 256, (4,4)),nn.ReLU(), # adaptive average pooling layer
            )
            self.fc = nn.Sequential(
            nn.Linear(9216, 4096),nn.Sigmoid()  # instead of adaptive average pooling
            )
            self.out = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid()) # fully connected layer with 1 neuron
            # Initialize the weights and biases of the layers using the method described in the paper
            self.method = "paper"
            self.conv.apply(self.init_weights_paper)
            self.fc.apply(self.init_weights_paper)
            self.out.apply(self.init_weights_paper)
        else:
            if arch == "arch_1":
                self.arch = "arch_1"
                self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=10, stride=2, padding=2), nn.LeakyReLU(0.1),nn.MaxPool2d(2),nn.Dropout(0.2),
                nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=2), nn.LeakyReLU(0.1),nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=2), nn.LeakyReLU(0.1),nn.MaxPool2d(2), 
                nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1), nn.LeakyReLU(0.1),
                )
                self.fc = nn.Sequential(
                    nn.Linear(6400, 4096), nn.Linear(4096, 2042),nn.ReLU()  # instead of adaptive average pooling
                )
                self.out = nn.Sequential(nn.Linear(2042, 1), nn.Sigmoid()) # fully connected layer with 1 neuron
            
            if arch == "arch_2":
                self.arch = "arch_2"
                self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=10, stride=2, padding=2), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=2), nn.LeakyReLU(0.1),nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=4, stride=1, padding = 1), nn.LeakyReLU(0.1),nn.MaxPool2d(2),  
                nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1), nn.LeakyReLU(0.1),
                nn.AdaptiveAvgPool2d((6,6))
                )
                self.fc = nn.Sequential(nn.Linear(6*6*256, 4096), nn.Sigmoid()) # fully connected layer with 4096 neurons
                self.out = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid()) # fully connected layer with 1 neuron
            
            # Initialize the weights and biases of the layers using the method described in the paper
            self.conv.apply(self.weights_init_xavier)
            self.fc.apply(self.weights_init_xavier)
            self.out.apply(self.weights_init_xavier)
            self.method = "xavier"
            
            
            
    def forward(self, img1, img2):
        """
        this function takes two images as input and returns a single value that indicates if the images are of the same
        person or not. The images are passed through the convolutional part of the network, then through the fully
        connected part, and finally through the output part.
        params:
            img1: torch.Tensor - The first image
            img2: torch.Tensor - The second image
        """
        
        input1 = self.conv(img1) # Pass the first image through the convolutional part
        input1 = input1.view(input1.shape[0], -1) # Flatten the output
        input1 = self.fc(input1)# Pass the output through the fully connected part

        input2 = self.conv(img2)# Pass the second image through the convolutional part
        input2 = input2.view(input2.shape[0], -1)# Flatten the output
        input2 = self.fc(input2)# Pass the output through the fully connected part

        res = self.out(torch.abs(input1-input2))# Pass the absolute difference of the two outputs through the output part

        return res

    def init_weights_paper(self, m):
        # Initialize the weights and biases of the layers using the method described in the paper
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            m.bias.data.normal_(0, 0.01)
      
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.2)
            m.bias.data.normal_(0, 0.1)
    
    def weights_init_xavier(self,m):
        # Initialize the weights and biases of the layers using the Xavier method
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data) # initialize the weights with xavier normal
            nn.init.normal_(m.bias.data) # initialize the bias with normal distribution
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data) # initialize the weights with xavier normal
            nn.init.normal_(m.bias.data) # initialize the bias with normal distribution

def train(data_loader,model,loss_fn,optimizer,epoch,device,writer,print_steps =True):
    """
    This function trains the model for one epoch. It takes the data loader, the model, the loss function, the optimizer,
    the epoch number, and the device as input. It returns the average loss and accuracy of the epoch.
    params: 
        data_loader: torch.utils.data.DataLoader - The data loader that contains the training data
        model: torch.nn.Module - The model that should be trained
        loss_fn: torch.nn.Module - The loss function that should be used
        optimizer: torch.optim.Optimizer - The optimizer that should be used
        epoch: int - The epoch number
        device: str - The device that should be used for training
        writer: torch.utils.tensorboard.SummaryWriter - The tensorboard writer that should be used
        print_steps: bool - If True, the function should print the average loss and accuracy every 5 steps
    """
    model.train() # Set the model to training mode
    # Initialize the total loss and accuracy
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    
    for step, (x1, x2, y) in enumerate(data_loader):
        x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)# Move the input tensors to the device   
        # No need to track the gradients
        #calculate the loss and accuracy for the current batch before the backward pass
        if epoch == 0 and step == 0:
            writer.add_graph(model, (x1[1], x2[1]))
        with torch.no_grad(): 
            pred = model(x1[1], x2[1])
            loss = loss_fn(pred, y)
            total_loss += loss.item() * y.size(0)  # Accumulate scaled loss
            preds = (pred > 0.5).float()
            acc = (preds == y).float().mean()
            total_acc += acc.item() * y.size(0)
            total_samples += y.size(0)
            if print_steps and (step + 1) % 5 == 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_acc / total_samples
                print(f'Step {step+1}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}')
                ##write the loss and accuracy to tensorboard
                writer.add_scalar('Loss/train_steps',loss.item(), epoch * len(data_loader) + step+1)
                writer.add_scalar('Accuracy/train_steps',acc.item(), epoch * len(data_loader) + step)
                #imgae grid for tensorboard #just one pair from the batch
                concatenated_images = torch.cat((x1[1][0:1], x2[1][0:1]), 0)
                # Create a grid of images: adjust nrow to control how many images per row
                image_grid = make_grid(concatenated_images * 0.5 + 0.5, nrow=4)  # Assuming nrow=4 for side-by-side comparison
                writer.add_image('training/image_pairs', image_grid, epoch * len(data_loader) + step+1)
                # Log the weights and biases of the model every 5 steps
                for name, param in model.named_parameters():
                    # Log histograms of weights
                    if 'weight' in name:
                        writer.add_histogram(f'{name}/weights', param.data.cpu().numpy(), epoch * len(data_loader) + step+1)
                    # Log histograms of biases
                    if 'bias' in name:
                        writer.add_histogram(f'{name}/biases', param.data.cpu().numpy(), epoch * len(data_loader) + step+1)
                    
        def closure(): # Define the closure function for the optimizer to perform L2 regularization and update the weights
            optimizer.zero_grad()
            # Forward pass
            pred = model(x1[1], x2[1])
            # Compute loss
            loss = loss_fn(pred, y)
            #L2 regularization
            l2_lambda = 0.0000001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss_with_l2 = loss + l2_lambda * l2_norm
            # Backward pass
            loss_with_l2.backward()
            return loss_with_l2
        optimizer.step(closure) # Perform the optimizer step
    

        
    avg_loss = total_loss / total_samples #calculate the average loss
    avg_acc = total_acc / total_samples   #calculate the average accuracy
    print(f'Epoch {epoch} Completed, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}')
    return avg_loss, avg_acc    
    
def test(dataloader, model, loss_fn, device, epoch, print_test,exp_name =None,for_report = False):
    """
    This function tests the model on the test data. It takes the data loader, the model, the loss function, the device,
    the epoch number, and a boolean that indicates if the test summary should be printed as input. It returns the
    accuracy of the test data.
    params:
        dataloader: torch.utils.data.DataLoader - The data loader that contains the test data
        model: torch.nn.Module - The model that should be tested
        loss_fn: torch.nn.Module - The loss function that should be used
        device: str - The device that should be used for testing
        epoch: int - The epoch number
        print_test: bool - If True, the function should print the test summary
        exp_name: str - The name of the experiment
        for_report: bool - If True, the function should save the confusion matrix and the examples of TP, TN, FP, and FN
        
    """
    model.eval() # Set the model to evaluation mode
    # Initialize the total loss and correct predictions
    total_loss, total_correct = 0, 0
    total_samples = 0
    if for_report:
        assert exp_name is not None, "The exp_name parameter should be defined"
        TP, TN, FP, FN = 0, 0, 0, 0 # initialize the counters for the confusion matrix
        TP_example_counter,TN_example_counter,FP_example_counter,FN_example_counter = 0,0,0,0 # initialize the counters for the right and wrong predictions

    #calculate the loss and accuracy for the test data
    with torch.no_grad():
        for step, (x1, x2, y) in enumerate(dataloader):
            # Assuming x1 and x2 are the input tensors that need to be moved to the device
            x1[1], x2[1], y = x1[1].to(device), x2[1].to(device), y.to(device)
            pred = model(x1[1], x2[1])
            total_loss += loss_fn(pred, y).item()
            preds = (pred > 0.5).float()  # Apply threshold to obtain binary predictions
            total_correct += (preds == y).float().sum().item()
            total_samples += y.size(0)
            if for_report:
                x1[1] = x1[1]*0.5 + 0.5  # denormalize the image for saving
                x2[1] = x2[1]*0.5 + 0.5 # denormalize the image for saving 
                for i in range(len(y)):
                    if y[i] == 1 and preds[i] == 1:
                        TP += 1
                        if TP_example_counter <=2:
                            path =f"{exp_name}/TP_example_{TP}" 
                            os.makedirs(path,exist_ok=True)
                            save_image(x1[1][i],os.path.join(path,f"TP_example_img_1_{i+1}_{x1[0][i]}.jpg"))
                            save_image(x2[1][i],os.path.join(path,f"TP_example_img_2_{i+1}_{x2[0][i]}.jpg"))
                            TP_example_counter += 1
                            
                    elif y[i] == 0 and preds[i] == 0:
                        TN += 1
                        if TN_example_counter <=2:
                            TN_example_counter += 1
                            path =f"{exp_name}/TN_examples_{TN_example_counter}" 
                            os.makedirs(path,exist_ok=True)
                            save_image(x1[1][i],os.path.join(path,f"TN_example_img_1_{i+1}_{x1[0][i]}.jpg"))
                            save_image(x2[1][i],os.path.join(path,f"TN_example_img_2_{i+1}_{x2[0][i]}.jpg"))
                            
                            
                    elif y[i] == 0 and preds[i] == 1:
                        FP += 1
                        if FP_example_counter <=2:
                            FP_example_counter += 1
                            path =f"{exp_name}/FP_examples_{FP_example_counter}" 
                            os.makedirs(path,exist_ok=True)
                            save_image(x1[1][i],os.path.join(path,f"FP_example_img_1_{i+1}_{x1[0][i]}.jpg"))
                            save_image(x2[1][i],os.path.join(path,f"FP_example_img_2_{i+1}_{x2[0][i]}.jpg"))
                            
                                                    
                    elif y[i] == 1 and preds[i] == 0:
                        FN += 1
                        if FN_example_counter <=2:
                            FN_example_counter += 1
                            path =f"{exp_name}/FN_examples_{FP_example_counter}" 
                            os.makedirs(path,exist_ok=True)
                            save_image(x1[1][i],os.path.join(path,f"FN_example_img_1_{i+1}_{x1[0][i]}.jpg"))
                            save_image(x2[1][i],os.path.join(path,f"FN_example_img_2_{i+1}_{x2[0][i]}.jpg"))
                                                  
        if for_report: #save the confusion matrix
            confusin_matrix = np.array([[TP,FP],[FN,TN]])
            save_confusion_matrix(confusin_matrix,exp_name)
    
    #calculate the average loss and accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    if print_test:#print the test summary
        print(f"---------------\nTest Summary\nEpoch: {epoch}, Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return accuracy,avg_loss

def save_confusion_matrix(confusion_matrix,exp_name):
    """
    save the confusion matrix as a heatmap
    params:
        confusion_matrix: np.array - The confusion matrix
        exp_name: str - The name of the experiment
        path : str - The path to save the confusion matrix
    """
    # Create a heatmap
    sns.set_theme(color_codes=True)
    plt.figure(1, figsize=(8, 6))
    
    plt.title('Confusion Matrix')
    labels = np.array([["TP", "FP"], ["FN", "TN"]])  # Label array corresponding to confusion matrix
    
    # Prepare to display both numbers and labels
    # Manually create combined labels with formatting
    combined_labels = [[f"{label}\n{value}" for label, value in zip(row_labels, row_values)] 
                       for row_labels, row_values in zip(labels, confusion_matrix)]
    
    sns.set_theme(font_scale=1.4)
    ax = sns.heatmap(confusion_matrix, annot=combined_labels, fmt='', cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    
    ax.set_xticklabels(['Same', 'Different'])
    ax.set_yticklabels(['Same', 'Different'])
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(os.path.join(f'{exp_name}',f'conf_matrix.png'), dpi=100)
    plt.close()
    
def weight_reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def run_experiment(train_loaders,test_loader,model,loss_fn,epochs,device,exp_name,lr = 1e-4):
    """
    This function runs the experiment. It takes the training data loaders, the test data loader, the model, the loss
    function, the optimizer, the number of epochs, the device, and the experiment name as input. It trains the model
    and tests it on the test data. 
    params:
        train_loaders: dict - A dictionary that contains the training data loaders
        test_loader: torch.utils.data.DataLoader - The data loader that contains the test data
        model: torch.nn.Module - The model that should be trained
        loss_fn: torch.nn.Module - The loss function that should be used
        optimizer: torch.optim.Optimizer - The optimizer that should be used
        epochs: int - The number of epochs
        device: str - The device that should be used for training
        exp_name: str - The name of the experiment
        lr: float - The learning rate that should be used for training the model default = 1e-4    
    """
    
    
   
    model.to(device) # Move the model to the device
    optimizer = torch.optim.Adam(model.parameters(), lr) # Initialize the optimizer
    for i,train_loader in enumerate(train_loaders.values()): # Train the model on each training data loader with and without augmentations
        exp = f"exp_{exp_name}/{list(train_loaders.keys())[i]}"
        writer = SummaryWriter(f'runs/{exp}')
        break_criteria = 0 #initialize the break criteria counter
        
        for epoch in tqdm(range(epochs),colour = 'green'):
            start_tiem = time.time()
            print("#####################################")
            print(f"Epoch {epoch+1}")
            avg_loss,avg_acc = train(train_loader, model, loss_fn, optimizer, epoch, device,writer)
            #log the average loss and accuracy to tensorboard for each epoch
            writer.add_scalar('Loss/train_Epoch_avg', avg_loss, epoch)
            writer.add_scalar('Accuracy/train_Epoch_avg', avg_acc, epoch)
            break_criteria += 1 if avg_acc > 0.9 else 0
            avg_acc_test,avg_loss_test= test(test_loader, model, loss_fn,device, epoch , print_test= True) #test the model on epoch
            #log the average loss and accuracy to tensorboard for each epoch
            writer.add_scalar('Loss/test_epoch_avg ', avg_loss_test, epoch)
            writer.add_scalar('Accuracy/test_epoch_avg', avg_acc_test, epoch)
            if break_criteria >= 3:
                break
            print("\n")
        end_time = time.time()
        print("Training is Done!")
        run_time = end_time - start_tiem
        hparms  = {
                "lr":lr,
                "epochs":epochs,
                "train batch_size":64,
                "optimizer":"Adam",
                "loss_function":"BCELoss",
                "architecture":model.arch,
                "weight_initialization":model.method,
                "augnentation": "without augmantaton" if train_loader.dataset.augmentations is False else "with augmentation",
                "run time":run_time,
                "total epochs to converge":epoch+1     
            }
        #log the hyperparameters and the average accuracy and loss of the tet after the trainig to tensorboard
        writer.add_hparams(hparms, {"hparam/final test accuracy":avg_acc_test,"hparam/final test loss":avg_loss_test})
        test(test_loader, model, loss_fn,device, epoch , print_test= False,exp_name = exp,for_report = True) #get the report figures    
        #reset the model for the next sub_experiment
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(weight_reset)
    print("The experiment is done!")
  

#%% Main code runing the experiments
#set the train and test data paths
img_dir_train = "/home/itamara/Desktop/project/pairsDevTrain.txt"
img_dir_test = "/home/itamara/Desktop/project/pairsDevTest.txt"

#Defining the hyperparameters, the device, the loss function, and the optimizer
epochs = 40
lr  = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.BCELoss()

#Defining the augmentations
augmentations = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Randomly rotate images in the range (-20, 20) degrees
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    
])

#Creating DataLoaders and Datasets for the training and test data 
img_set_train_with_aug = TwoImageDataset(img_dir_train, augmentations)
img_set_train_without_aug = TwoImageDataset(img_dir_train)
img_set_test = TwoImageDataset(img_dir_test)

train_loader_with_aug  = DataLoader(img_set_train_with_aug, batch_size=64, shuffle=True)
trian_loader_without_aug = DataLoader(img_set_train_without_aug, batch_size=64, shuffle=True)
test_loader = DataLoader(img_set_test, batch_size=64, shuffle=True)
#Creating a dictionary that contains the training data loaders with and without augmentation
train_loaders = {
    "with_aug": train_loader_with_aug,
    "without_aug": trian_loader_without_aug
   
    }
   


#build the different models
model_arch_paper = SiameseNN(arch = 'paper')
model_arch_1 = SiameseNN(arch = 'arch_1')
model_arch_2 = SiameseNN(arch = 'arch_2')



#run the experiments
run_experiment(train_loaders,test_loader,model_arch_2,loss_fn,epochs,device,"arch_2",lr = 1e-4)
run_experiment(train_loaders,test_loader,model_arch_1,loss_fn,epochs,device,"arch_1",lr = 1e-4)
run_experiment(train_loaders,test_loader,model_arch_paper,loss_fn,epochs,device,"paper_arch",lr = 1e-4)



