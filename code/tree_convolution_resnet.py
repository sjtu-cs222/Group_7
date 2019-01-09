import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from weight_library import *
import numpy as np
from torch.utils.data.dataloader import DataLoader
from dataset_sampler import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from DML import *
import copy

class TreePruning(nn.Module):
    #currently: supporting sequential. No skip connection.
    def __init__(self, model, big_sampler, train_loader, val_loader, test_loader, epoches_list = None, num_classes=10):
        super(TreePruning, self).__init__()
        self.model = model.cpu()
        self.epoches_list = epoches_list
        self.big_sampler = big_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.wl = WeightLibrary()
        self.wl.save_library(model)
        self.layer_groupings = []
        self.group_num_lis = []
        self.precisions = []
        #+1 because the descendents of the final layer buckets are themselves(for simplicity)
        self.descendents = []
        self.pretrain_optimizer = optim.SGD(params = self.model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        self.pretrain_scheduler = lr_scheduler.StepLR(self.pretrain_optimizer, step_size=30, gamma=0.5)
        self.optimizer = optim.SGD(params = self.model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=True)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def gen_groupings(self, group_num_lis):
        self.group_num_lis = group_num_lis
        #Assume the convolutional layers come earlier.
        for i in range(len(self.wl.conv_names)):
            layer_name = self.wl.conv_names[i]
            #print(layer_name)
            self.layer_groupings.append(self.gen_grouping(layer_name, group_num_lis[i]))
            self.descendents.append([])
            self.precisions.append([])
        
        for i in range(len(self.wl.fc_names)):
            layer_name = self.wl.fc_names[i]
            self.layer_groupings.append(self.gen_grouping(layer_name, group_num_lis[i+len(self.wl.conv_names)]))
            self.descendents.append([])
            self.precisions.append([])
        
        last_descendents = []
        #Each output class is a group!
        last_groupings = [[i] for i in range(self.num_classes)]
        last_groupings = np.array(last_groupings)
        
        for i in range(self.num_classes):
            base_set = set()
            base_set.add(i)
            last_descendents.append(base_set)
        self.descendents.append(last_descendents)
        self.layer_groupings.append(last_groupings)
    
    def gen_grouping(self, layer_name, group_num=8):
        if layer_name in self.wl.conv_names or layer_name in self.wl.fc_names:
            filter_size = self.wl.state_dict[layer_name].shape
            conv_channels = filter_size[1]
            #print(filter_size)
            if conv_channels == 3:
                #input
                return None
            else:
                groups = []
                group_size = (conv_channels+group_num-1) // group_num
                permutation_idx = np.random.permutation(conv_channels)
                for i in range(group_num):
                    groups.append(permutation_idx[group_size*i:group_size*(i+1)])
                
                return np.array(groups)
    
    #To see whether no training works?
    def train_all(self, max_epoch=0):
        self.model.load_state_dict(torch.load('model_res164.pkl'))
        self.wl.save_library(self.model)
        
        if self.epoches_list is None:
            for idx in range(self.wl.LAYERS-1, -1, -1):
                self.train_layer(idx, max_epoch)
        else:
            for idx in range(self.wl.LAYERS-1, -1, -1):
                self.train_layer(idx, self.epoches_list[idx])
        
    def train_layer(self, idx, max_epoch=1):
            if idx >= self.wl.LAYERS:
                print('Index out of range!')
                return
            if idx <= len(self.wl.conv_names)-1:
                layer_name = self.wl.conv_names[idx]
            else:
                layer_name = self.wl.fc_names[idx-len(self.wl.conv_names)]
            
            groups = self.layer_groupings[idx]
            if groups is None:
                #Problem here: first convolution layer. Perhaps we don't prune it.
                return
            #TBD: overflow
            SKIP_FLAG = 0
            if idx+1 <= len(self.wl.conv_names)-1:
                next_layer_name = self.wl.conv_names[idx+1]
            else:
                next_layer_name = ""
            
            next_groups = self.layer_groupings[idx+1]
            
            
            group_num = len(groups)
            next_group_num = len(next_groups)
            if idx <= len(self.wl.conv_names)-1:
                layer_filter_size = self.wl.conv_shapes[idx]
            else:
                layer_filter_size = self.wl.fc_shapes[idx-len(self.wl.conv_names)]
            
            loaded_filter = self.wl.load_weights(idx)
            
            #entry i,j of the table means when bucket j in previous layer is connected with bucket i in the next layer,
            #the average accuracy of all classes descendent from bucket i(next layer) is this value in the table.
            connection_accuracy_table = np.zeros((next_group_num, group_num))
            
            for i in range(group_num):
                print('**********training group #%d**********'%(i+1))
                filter_mask = torch.zeros(layer_filter_size).float()
                if len(layer_filter_size)==4:
                    filter_mask[:,groups[i],:,:] = 1.0
                else:
                    filter_mask[:,groups[i]] = 1.0
                
                #fix?
                loaded_filter_masked = loaded_filter*filter_mask
                #need to be careful about where to put cuda
                #Step1. Mask
                copied_state_dict = self.model.state_dict().copy()
                copied_state_dict[layer_name] = loaded_filter_masked.cuda()
                self.model.load_state_dict(copied_state_dict)
                #connect only one path, predict the accuracy
                #Step2. Train the model, when one bucket in layer i is fully connected to layer i+1.
                self.train_layer_one_connection(idx, max_epoch, filter_mask)
                #evaluate all classes, get the accuracies of all classes.
                #Step3.1 Predict the accuracy of all classes
                accs = self.evaluate_layer(idx)
                #print(accs)
                #Step3.2 Pick out the descendent classes for each bucket in layer i+1
                up_b = len(self.descendents[idx+1]) if SKIP_FLAG == 0 else len(self.descendents[idx+2])
                for k in range(up_b):
                    descendent_set = self.descendents[idx+1][k] if SKIP_FLAG == 0 else self.descendents[idx+2][k]
                    total_acc = 0.0
                    mean_acc = 0.0
                    for x in descendent_set:
                        total_acc += accs[x]
                    if total_acc > 0.0:
                        mean_acc = total_acc / len(descendent_set)
                    else:
                        mean_acc = 0.0
                    #Step3.3 Fill in column i for the connection-avg. acc table.
                    connection_accuracy_table[k][i] = mean_acc
                #Step3.4 Update the weights
                if len(layer_filter_size)==4:
                    loaded_filter[:,groups[i],:,:] = self.model.state_dict()[layer_name][:,groups[i],:,:].cpu()
                else:
                    loaded_filter[:,groups[i]] = self.model.state_dict()[layer_name][:,groups[i]].cpu()
            
            final_filter = torch.zeros(loaded_filter.shape)
            #Step4 Choose connections.
            best_connections = np.argmax(connection_accuracy_table, 1)
            print('Best connections:', best_connections)
            print(connection_accuracy_table)
            
            #Step5 Update descendent information.
            layer_descendent = [[] for i in range(group_num)]
            for i in range(next_group_num):
                layer_descendent[best_connections[i]].append(i)
                
            for i in range(group_num):
                base_set = set()
                for x in layer_descendent[i]:
                    if not SKIP_FLAG:
                        base_set = base_set.union(self.descendents[idx+1][x])
                    else:
                        base_set = base_set.union(self.descendents[idx+2][x])
                self.descendents[idx].append(base_set)
            
            #Step6 Reconnect
            for i in range(len(best_connections)):
                #Eliminate connections not in the graph
                if not SKIP_FLAG and len(self.descendents[idx+1][i]) == 0:
                    continue
                elif SKIP_FLAG and len(self.descendents[idx+2][i]) == 0:
                    continue
                if len(layer_filter_size)==4:
                    #conv
                    for k in next_groups[i]:
                        final_filter[k,groups[best_connections[i]],:,:] = loaded_filter[k,groups[best_connections[i]],:,:]
                else:
                    #fc
                    for k in next_groups[i]:
                        final_filter[k,groups[best_connections[i]]] = loaded_filter[k,groups[best_connections[i]]]
            
            self.wl.save_weights(idx, final_filter)
            
            #REALLY save connections
            copied_state_dict = self.model.state_dict().copy()
            copied_state_dict[layer_name] = final_filter.cuda()
            self.model.load_state_dict(copied_state_dict)
            
            torch.save(self.model.state_dict(), 'checkpoints/model_layer_%d.pkl'%idx)
            print('Evaluating after layer %d is done'%idx)
            self.evaluate_test()
            
            print('Sparsity:%d/%d'%(np.sum(final_filter.numpy()!=0), final_filter.numel()))
            print('Connection:', self.descendents[idx])
            
    def train_layer_one_connection(self, idx, max_epoch=2, filter_mask=None):
        layer_type = ""
        if idx >= len(self.wl.conv_names):
            layer_name = self.wl.fc_names[idx-len(self.wl.conv_names)]
            layer_type = "fc"
        else:
            layer_name = self.wl.conv_names[idx]
            layer_type = "conv"
        
        #Should train conv?
        #if layer_type == 'conv':
        #    max_epoch = 1
        
        print("Training layer %s"%layer_name)
        #VGG
        if self.model.net_type == 'Network':
            if layer_type == "conv":
                if not "downsample" in layer_name:
                    params = getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]).parameters()
                else:
                    params = getattr(getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]), '0').parameters()
            else:
                params = self.model.model.fc.parameters()
        
        self.optimizer = optim.SGD(params=params, lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=True)
        self.model = self.model.cuda()
        self.model.train()
        
        self.evaluate_layer(idx)
        self.evaluate_test()
        
        max_epoch = 1 if max_epoch > 0 else 0
        for epoch in range(max_epoch):
            for idx_, (data, target) in enumerate(self.train_loader):
                if idx_ >= 100:
                    break
                data, target = data.cuda(), target.cuda()
                prediction = self.model(data)
                pred_classes = prediction.max(1)[1]
                loss = self.loss_fn(prediction, target) 
                acc = (torch.sum(pred_classes==target)).cpu().numpy().astype(np.float32)/prediction.size(0)
                self.optimizer.zero_grad()
                loss.backward()            
                
                if layer_type == 'fc':
                    self.model.model.fc.weight.grad *= filter_mask.cuda()
                else:
                    #self.model.model.features?
                    if not "downsample" in layer_name:
                            getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]).weight.grad *= filter_mask.cuda()
                    else:
                        getattr(getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]), '0').weight.grad *= filter_mask.cuda()
                        
                self.optimizer.step()
                
                if idx_ % 20 == 0:
                    print('EPOCH %d, IDX %d, LOSS %.4f, ACC %.4f'%(epoch+1, idx_+1, loss.item(), acc.item()))
            
            self.evaluate_layer(idx)
            self.evaluate_test()
            self.big_sampler.shuffle()
            self.train_loader = DataLoader(trainset, sampler=self.big_sampler.train_sampler,batch_size=64)
            
        return
    
    def pretrain(self, max_epoch=160):
        self.model = self.model.cuda()
        self.model.train()
        cur_acc = 0
        for epoch in range(max_epoch):
            self.pretrain_scheduler.step()
            for idx, (data, target) in enumerate(self.train_loader):
                data, target = data.cuda(), target.cuda()
                prediction = self.model(data)
                pred_classes = prediction.max(1)[1]
                loss = self.loss_fn(prediction, target)
                acc = (torch.sum(pred_classes==target)).cpu().numpy().astype(np.float32)/prediction.size(0)
                self.pretrain_optimizer.zero_grad()
                loss.backward()
                self.pretrain_optimizer.step()
                if idx % 20 == 0:
                    print('EPOCH %d, IDX %d, LOSS %.4f, ACC %.4f'%(epoch+1, idx+1, loss.item(), acc.item()))
            acc = self.evaluate_test()
            if acc > cur_acc:
                cur_acc = acc
                torch.save(self.model.state_dict(), 'model_vgg19bn.pkl')
                print('Current highest accuracy: %.4f.'%cur_acc)
           
            self.big_sampler.shuffle()
            self.train_loader = DataLoader(trainset, sampler=self.big_sampler.train_sampler,batch_size=64)
        
        
        
    def evaluate_layer(self, idx):
        #print(idx)
        if idx >= len(self.wl.conv_names):
            layer_name = self.wl.fc_names[idx-len(self.wl.conv_names)]
        else:
            layer_name = self.wl.conv_names[idx]
            
        filt = self.model.state_dict()[layer_name].cpu().numpy()
        print('Nonzero parameters:', np.sum(filt!=0))
        
        self.model.eval()
        total_correct = 0
        total_seen = 0
        total_correct_classes = np.zeros(self.num_classes).astype(np.int)
        total_seen_classes = np.zeros(self.num_classes).astype(np.int)
        for idx, (data, target) in enumerate(self.val_loader):
            total_seen += data.size(0)
            data, target = data.cuda(), target.cuda()
            prediction = self.model(data)
            pred_classes = prediction.max(1)[1]
            total_correct += torch.sum(pred_classes==target).item()
            
            eq_inf = (pred_classes == target).cpu().numpy()
            class_cnt = np.bincount(target.cpu().numpy(), minlength=self.num_classes)
            total_seen_classes += class_cnt
            pseudo_target = target.cpu().numpy()
            #A trick, this class doesn't exist.
            pseudo_target[np.where(eq_inf!=1)[0]] = self.num_classes + 1
            true_cnt = np.bincount(pseudo_target, minlength=self.num_classes)[:self.num_classes]
            total_correct_classes += true_cnt
        
        print('total seen:', total_seen_classes)
        print('total correct:', total_correct_classes)
        print(float(total_correct)/total_seen)
        self.model.train()
                    
        #accs = []
        #test data
        accs = total_correct_classes / total_seen_classes
        print('accuracy:', accs)
        return accs
    
    def evaluate_test(self):
        self.model.eval()
        total_correct = 0
        total_seen = 0
        for idx, (data, target) in enumerate(self.test_loader):
            total_seen += data.size(0)
            data, target = data.cuda(), target.cuda()
            prediction = self.model(data)
            pred_classes = prediction.max(1)[1]
            total_correct += torch.sum(pred_classes==target).item()
        acc = float(total_correct)/total_seen
        print(acc)
        self.model.train()
        return acc
    
    def evaluate_val(self):
        self.model.eval()
        total_correct = 0
        total_seen = 0
        for idx, (data, target) in enumerate(self.val_loader):
            total_seen += data.size(0)
            data, target = data.cuda(), target.cuda()
            prediction = self.model(data)
            pred_classes = prediction.max(1)[1]
            total_correct += torch.sum(pred_classes==target).item()
        print(float(total_correct)/total_seen)
        self.model.train()
        
    def retrain(self, max_epoch):
        
        def gen_filters():
            model_copy = copy.deepcopy(self.model)
            filters = {}
            for layer_name in model_copy.state_dict():
                    if layer_name in self.wl.fc_names:
                        filter_np = model_copy.model.fc.weight.cpu().detach().numpy()
                        filter_mask = torch.from_numpy((filter_np!=0).astype(np.float32))
                        filters[layer_name] = filter_mask
                        
                    elif layer_name in self.wl.conv_names:
                        if 'layer' in layer_name:
                            if not "downsample" in layer_name:
                                filter_np =  getattr(getattr(getattr(model_copy.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]).weight.cpu().detach().numpy()
                            else:
                                filter_np = getattr(getattr(getattr(getattr(model_copy.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]), '0').weight.cpu().detach().numpy()
                        else:
                            filter_np = model_copy.model.conv1.weight.cpu().detach().numpy()
                        filter_mask = torch.from_numpy((filter_np!=0).astype(np.float32))
                        filters[layer_name] = filter_mask
            return filters
        
        cur_acc = 0
        
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load('checkpoints/model_layer_115.pkl'))
        
        self.optimizer = optim.SGD(params = self.model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True)
        #self.optimizer = optim.Adam(params = self.model.parameters(), lr=0.003, weight_decay=1e-4, betas=(0.9,0.999))
        self.scheduler = lr_scheduler.StepLR(self.pretrain_optimizer, step_size=30, gamma=0.5)
        filter_masks = gen_filters()
        
        #Add deep mutual learning
        """
        criterion_kl = KL_loss()
        big_model = Network()
        big_model.load_state_dict(torch.load('model.pkl'))
        big_model = big_model.cuda()
        big_model_optimizer = optim.SGD(params = big_model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9, nesterov=True)
        big_model_scheduler = lr_scheduler.StepLR(big_model_optimizer, step_size=30, gamma=0.5)
        """
        for epoch in range(max_epoch):
            #big_model_scheduler.step()
            self.scheduler.step()
            for idx, (data, target) in enumerate(self.train_loader):
                data, target = data.cuda(), target.cuda()
                prediction = self.model(data)
                pred_classes = prediction.max(1)[1]
                
                #big_model_prediction = big_model(data)
                
                loss = self.loss_fn(prediction, target)# + criterion_kl(big_model_prediction, prediction)
                acc = (torch.sum(pred_classes==target)).cpu().numpy().astype(np.float32)/prediction.size(0)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                for layer_name in self.model.state_dict():
                    if layer_name in filter_masks.keys():
                        filter_mask = filter_masks[layer_name]
                        if layer_name in self.wl.conv_names:
                            if 'layer' in layer_name:
                                if not "downsample" in layer_name:
                                    getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]).weight.grad *= filter_mask.cuda()
                                else:
                                    getattr(getattr(getattr(getattr(self.model.model, layer_name.split('.')[1]), layer_name.split('.')[2]), layer_name.split('.')[3]), '0').weight.grad *= filter_mask.cuda()
                            else:
                                self.model.model.conv1.weight.grad *= filter_mask.cuda()
                        else:
                            self.model.model.fc.weight.grad *= filter_mask.cuda()                
                    
                self.optimizer.step()
                
                """
                loss_big = self.loss_fn(big_model_prediction, target) + criterion_kl(prediction, big_model_prediction)
                big_model_optimizer.zero_grad()
                loss_big.backward()
                big_model_optimizer.step()
                """
                
                if idx % 20 == 0:
                    print('EPOCH %d, IDX %d, LOSS %.4f, ACC %.4f'%(epoch+1, idx+1, loss.item(), acc.item()))
            
            print('Evaluating on validation set:')
            self.evaluate_val()
            print('Evaluating on test set:')
            acc = self.evaluate_test()
            
            if acc > cur_acc:
                cur_acc = acc
                torch.save(self.model.state_dict(), 'checkpoints/model_retrain.pkl')
                print('Current best accuracy is %.4f.'%acc)
            
            self.big_sampler.shuffle()
            self.train_loader = DataLoader(trainset, sampler=self.big_sampler.train_sampler,batch_size=64)
        
        

        
    
if __name__ == '__main__':
    from model import *
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='/home/tang/machine-learning/CS222-project/data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='/home/tang/machine-learning/CS222-project/data', train=False, download=False, transform=test_transform)
    n_training_samples =45000
    n_val_samples = 5000
    sampler = BigSampler(n_training_samples, n_val_samples)
    train_loader = DataLoader(trainset, sampler=sampler.train_sampler,batch_size=64)
    val_loader = DataLoader(trainset, sampler=sampler.val_sampler,batch_size=128)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    model = Network("Network").cuda()
    wl = WeightLibrary()
    wl.save_library(model)
    print(wl.conv_names)
    
    train_epoches = [1 for i in range(wl.LAYERS-1)] + [0]
    #train_epoches = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0]
    #train_epoches = [4,4,4,4,4,4,4,2,2,2,2,1,1,0]
    tree_pruning = TreePruning(model, sampler, train_loader, val_loader, test_loader, train_epoches)
    
    
    res_num_buckets = [0 for i in range(wl.LAYERS)]
    for i in range(len(res_num_buckets)):
        if i <= len(res_num_buckets)//4:
            res_num_buckets[i] = 1
        elif i <= 3 * len(res_num_buckets) // 4:
            res_num_buckets[i] = 2
        else:
            res_num_buckets[i] = 4
    #vgg_num_buckets = [1,2,2,2,4,4,4,4,4,8,8,8,8,16,16,16,16]
    #vgg_num_buckets = [1,1,1,2,2,4,4,4,4,8,8,8,8]
    tree_pruning.gen_groupings(res_num_buckets)
    layer_groupings = tree_pruning.layer_groupings
    print(len(layer_groupings))

    #tree_pruning.pretrain()
    
    #tree_pruning.train_all()
    #print(tree_pruning.descendents)
    
    tree_pruning.retrain(160)
    #Currently I think layerwise retraining will not work.
    #tree_pruning.retrain_layerwise(10)
    