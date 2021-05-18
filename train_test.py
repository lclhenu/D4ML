from torch.utils.data import Dataset, DataLoader, sampler
import time
import datetime
from dataset import *
from train_net import *
from loss import *
import random

'''
使用Github上原始的模型，但数据集等做了改变。
'''
test_acc_list = []
# from pytorchtools import EarlyStopping
nowtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
evaluate_flag = True
main_num_epochs = 60
pretrain_num_epochs = 80
# 是否使用在本案例集上当前relation之外的子集上预训练的模型
use_pretrain = False
batch_size = 64
learning_rate = 0.00025  # 0.00025
num_workers = 4
warm_epoch = 50
image_dim = 64
# 数据集的路径
# meta_data_path = 'UB/UBKinFace/meta_data/'
# meta_data_path = 'CK/CornellKinFace/meta_data/'
# meta_data_path = 'kii/KinFaceW-II/meta_data/'
meta_data_path = 'ki/KinFaceW-I/meta_data/'

strlist = meta_data_path.split('/')
data_name = strlist[1]

# image_path = 'UB/UBKinFace/images'
# image_path = 'CK/CornellKinFace/images'
# image_path = 'kii/KinFaceW-II/images'
image_path = 'ki/KinFaceW-I/images'

relationlist = ['fs', 'fd']
# relationlist = ['ms', 'md']
# relationlist = ['all']  # ck
# relationlist = ['set1']  # ub
# relationlist = ['set2'] # ub
data_transforms = {
    'train': transforms.Compose
    ([transforms.Resize([64, 64]), transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ]),
    'val': transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def getfoldnum_lenth(relation, meta_data_path):
    meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
    # meta_data['pairs'][:, 0]表示取meta_data['pairs']的所有行的第0个数据
    fold_allset = [d for d in meta_data['pairs'][:, 0]]
    num_sample = len(fold_allset)
    foldnum = len(np.unique(fold_allset))
    fold_allset = np.asarray(fold_allset)
    fold_uniqueset = np.unique(fold_allset)
    return foldnum, num_sample, fold_uniqueset, fold_allset


def train_test_split_postive(relation, meta_data_path, length, fold_allset, fold, test_size=0.2, shuffle=False,

                             random_seed=0):
    meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
    indices = list(range(0, length))
    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)
    if test_size != 0:  # 原来的方法其实是有问题的
        if type(test_size) is float and test_size != 0:
            split = int(test_size * length)
        elif type(test_size) is int and test_size != 0:
            split = test_size
        else:
            raise ValueError('parameter test_size should be an int or a float')
        return indices[split:], indices[:split], None  # 训练和测试数据的索引
    else:
        if fold_allset is not None:
            # # KI KII
            if data_name == 'KinFaceW-I' or data_name == 'KinFaceW-II':
                list_tr = [index for index, value in enumerate(fold_allset) if
                           value != fold]
                list_test = [index for index, value in enumerate(fold_allset) if
                             value == fold]
            # ck ub
            if data_name == 'UBKinFace' or data_name == 'CornellKinFace':
                list_tr = [index for index, value in enumerate(fold_allset) if
                           value != fold and meta_data['pairs'][index, 1] == 1]
                list_test = [index for index, value in enumerate(fold_allset) if
                             value == fold and meta_data['pairs'][index, 1] == 1]

            meta_data = sio.loadmat(os.path.join(meta_data_path, relation + '_pairs.mat'))
            return [list_tr, list_test]
        else:
            raise ValueError('% parameter \'fold_allset\' should be none when test_size is zero')


def loadtraintest(nsamples, image_path, meta_data_path, relation, data_transforms, fold_allset, aug, fold,
                  test_split=0.2,
                  batch_size=32, test=False):
    test_split = 0

    train_idx, test_idx = train_test_split_postive(relation, meta_data_path, nsamples, fold_allset, fold, test_split,
                                                   shuffle=False)  # shuffle表示是否需要随机取样本

    if not test:
        x_train = KinShipDataSetZL(relation, 'train', image_path, meta_data_path, train_idx, None, test_idx, fold,
                                   data_transforms['train'],
                                   aug=aug)

        train_loader = DataLoader(x_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

        dataloaders = {'train': train_loader}
        dataset_sizes = {'train': len(x_train)}

    else:
        x_test = KinShipDataSetZL(relation, 'test', image_path, meta_data_path, train_idx, None, test_idx, fold,
                                  data_transforms['test'],
                                  aug=False)
        test_loader = DataLoader(x_test, batch_size=batch_size)
        dataloaders = {'test': test_loader}
        dataset_sizes = {'test': len(x_test)}
    print(dataset_sizes)
    return dataloaders, dataset_sizes, train_idx, test_idx


class AdjustVariable(object):

    def __init__(self, name, start=0.9, stop=0.999, num_epochs=30):
        self.name = name
        self.start, self.stop = start, stop
        self.num_epochs = num_epochs
        self.ls = None

    def __call__(self, epoch):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, self.num_epochs)
        new_value = float(self.ls[epoch])
        return new_value


def euclidean_dist(x, y, label):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    all_negative_list = []
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    if dist.device != 'cpu':
        a_1 = dist.cpu().numpy().tolist()
    else:
        a_1 = dist.cpu().numpy().tolist()
    last = []
    negative_list = []
    weight = 0.0
    for i in range(dist.size(0)):
        used = []
        # negative_list = []
        b_1 = a_1[i][:]
        c_1 = b_1[:]
        del c_1[i]
        c_1.sort()
        c_1.reverse()
        while label[b_1.index(c_1[int((len(c_1) - 1) * weight)])] in last:
            if len(c_1) > 1:
                c_1.remove(c_1[int((len(c_1) - 1) * weight)])
                c_1.sort()
                c_1.reverse()
            else:
                break
        negative_list.append(b_1.index(c_1[int((len(c_1) - 1) * weight)]))
        last.append(label[b_1.index(c_1[int((len(c_1) - 1) * weight)])])
        # for j in range(0, len(c_1), int(weight)):
        #     d_1 = c_1[:]
        #
        #     # while [label[b_1.index(d_1[j])]] in used:
        #     #     if len(d_1) > int(len(c_1) / int(weight)):
        #     #         d_1.remove(d_1[j])
        #     #         d_1.sort()
        #     #         d_1.reverse()
        #     #     else:
        #     #         break
        #     negative_list.append(b_1.index(d_1[j]))
        #     last.append(label[b_1.index(d_1[j])])
        #     used.append(last)
        # all_negative_list.append(negative_list)

    return dist, negative_list


def euclidean_dist_test(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    all_negative_list = []
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    if dist.device != 'cpu':
        a_1 = dist.cpu().numpy().tolist()
    else:
        a_1 = dist.cpu().numpy().tolist()

    negative_list = []
    weight = 0.5
    for i in range(dist.size(0)):
        b_1 = a_1[i][:]
        c_1 = b_1[:]
        del c_1[i]
        c_1.sort()
        c_1.reverse()
        negative_list.append(b_1.index(c_1[int((len(c_1) - 1) * weight)]))
    return dist, negative_list


def train_model_no_val(model, dataloaders, criterion, criterion2, criterion3, criterion4, optimizer, fold, relation,
                       test_loader, StepLR, scheduler=None, linear_momentum=False, early_stop=False, ):
    since = time.time()
    num_epochs = model.num_epochs
    adjust_momentum = AdjustVariable('momentum', 0.9, 0.999, num_epochs)
    best_acc = .0
    best_test_acc = .0
    best_epoch = 0
    pth_name = '-1'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        total = 0
        correct = 0
        total_test = 0
        correct_test = 0
        if scheduler:
            scheduler.step()
        if linear_momentum:
            new_momentum = adjust_momentum(epoch)
            for param_groups in optimizer.param_groups:
                param_groups['momentum'] = new_momentum
        running_loss = 0.0
        model.train(True)

        for data in dataloaders['train']:
            gt = data['label'].float().to(device)
            gt_all = gt.view(-1, 1)  # KI KII
            image1_label = data['image1_label'].view(1, -1).squeeze(0).to(device)
            image2_label = data['image2_label'].view(1, -1).squeeze(0).to(device)
            input1 = data['input1'].to(device)  # 16, 3, 64, 64
            input2 = data['input2'].to(device)  # 16, 3, 64, 64
            image_label = torch.cat((image1_label, image2_label), dim=0)  # KI KII

            # ck ub
            if data_name == 'UBKinFace' or data_name == 'CornellKinFace':
                gt_all = torch.cat((gt, 0.0 * gt), dim=0).float().view(-1, 1)  # ck
                with torch.no_grad():  # 202.10.12
                    model.eval()
                    outputs1, feature_1positive, feature_2positive, f_id1, y1 = model(input1, input2)

                q = feature_1positive.view(feature_1positive.size(0), -1).float()
                w = feature_2positive.view(feature_2positive.size(0), -1).float()
                dist1, negative = euclidean_dist(q, w, image2_label)
                negative_sample = input2[negative[0]].unsqueeze(0)
                l1 = image2_label[negative[0]].unsqueeze(0)
                for i in range(1, len(negative)):
                    negative_sample = torch.cat((negative_sample, input2[negative[i]].unsqueeze(0)), dim=0)
                    l1 = torch.cat((l1, image2_label[negative[i]].unsqueeze(0)), dim=0)
                input1 = torch.cat((input1, input1), dim=0)
                input2 = torch.cat((input2, negative_sample), dim=0)
                image_label = torch.cat(
                    (torch.cat((image1_label, image1_label), dim=0), torch.cat((image2_label, l1), dim=0)), dim=0)
            # end ck ub

            model.train()
            outputs1, feature1, feature2, f_id, y = model(input1, input2)  # 全部
            feature1 = feature1.view(feature1.size(0), -1)
            feature2 = feature2.view(feature2.size(0), -1)
            f1 = feature1
            f2 = feature2
            target = image_label
            f = torch.cat((f1, f2), dim=0)
            p1 = (feature1 * gt_all)
            p2 = (feature2 * gt_all)
            n1 = feature1 * (1.0 - gt_all)
            n2 = feature2 * (1.0 - gt_all)
            another_gt = gt_all
            simi = torch.cosine_similarity(feature1, feature2, dim=1).view(-1, 1)
            preds = outputs1 > 0.5
            correct += float(torch.sum(preds == (gt_all > 0.5)))
            total += float(gt_all.size(0))
            loss1 = criterion(outputs1, gt_all)
            loss2 = criterion2(p1, p2)  # 正样本特征距离 /omega1 0.5
            loss3 = criterion2(n1, n2)  # 负样本特征距离 /omega2 0.1
            loss4 = criterion2(simi, another_gt)  # 余弦相似度
            loss5 = criterion3(f, target)  # 三元损失 /alpha 0.5
            part = {}
            num_part = 3
            for i in range(num_part):
                part[i] = y[i]
            loss6 = criterion4(part[0], image_label)
            for i in range(num_part - 1):
                loss6 += criterion4(part[i + 1], image_label)
            loss = loss1 + (loss6 / 3.0) + loss5 + 0.6 * loss3 + 5 * loss2 + loss4    # all
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders['train'])

        train_acc = 100.0 * correct / total
        print('train_acc:', float('%.2f' % train_acc))
        print('loss:', epoch_loss)
        for data in test_loader:
            input1 = data['input1'].to(device)
            input2 = data['input2'].to(device)
            gt = data['label'].float().to(device)
            gt_all = gt.view(-1, 1)  # KI KII
            # begin ck ub
            if data_name == 'UBKinFace' or data_name == 'CornellKinFace':
                gt = data['label'].to(device)
                gt_all = torch.cat((gt, 0.0 * gt), dim=0)
                q = input1.view(input1.size(0), -1).float()
                w = input2.view(input2.size(0), -1).float()
                dist1, negative = euclidean_dist_test(q, w)
                input_negative = input2[negative[0]].unsqueeze(0)
                for i in range(1, len(negative)):
                    input_negative = torch.cat((input_negative, input2[negative[i]].unsqueeze(0)), dim=0)
                input1 = torch.cat((input1, input1), dim=0)
                input2 = torch.cat((input2, input_negative), dim=0)
            # end ck ub

            with torch.no_grad():
                model.eval()
                outputs1, feature1, feature2, f_id, y = model(input1, input2)
                preds = outputs1 > 0.5
            total_test += float(gt_all.size(0))
            correct_test += float(torch.sum(preds == (gt_all > 0.5)))
        test_acc = 100.0 * correct_test / total_test
        print('test_acc:', '%.2f' % test_acc)

        if test_acc > best_acc:
            best_acc = float('%.2f' % test_acc)
            best_epoch = epoch
            best_test_acc = test_acc
            if os.path.exists(os.path.join('weights', relation + '_' + str(fold) + '_' + pth_name + '.pth')):
                os.remove(os.path.join('weights', relation + '_' + str(fold) + '_' + pth_name + '.pth'))
                pth_name = str('%.2f' % test_acc)
                torch.save(model.state_dict(),
                           os.path.join('weights', relation + '_' + str(fold) + '_' + pth_name + '.pth'))
            else:
                pth_name = str('%.2f' % test_acc)
                torch.save(model.state_dict(),
                           os.path.join('weights', relation + '_' + str(fold) + '_' + pth_name + '.pth'))
    test_acc_list.append(best_test_acc)
    print('best test acc:', best_test_acc, 'best epoch:', best_epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model


def mainonlytest():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for relation in relationlist:
            print('starting ', data_name + '#' + relation)
            # 首先获取在当前relation关系之外的数据集上训练的模型
            print('pretraining.....')
            manualSeed = 999
            print("Random Seed: ", manualSeed)
            print('The learning rate is ', learning_rate)
            print(strlist[1], '#', relation)
            random.seed(manualSeed)
            torch.manual_seed(manualSeed)
            global model  # 在函数内部改变函数外部的变量，就可以通过在函数内部声明变量为global变量
            foldnum, num_sample, fold_uniqueset, fold_allset = getfoldnum_lenth(relation, meta_data_path)
            for fold in range(1, foldnum + 1):
                print('fold=', fold)
                dataloaders, dataset_sizes, train_idx, test_idx = loadtraintest(num_sample, image_path, meta_data_path,
                                                                                relation, data_transforms,
                                                                                fold_allset, aug=False, fold=fold,
                                                                                test_split=0.03,
                                                                                batch_size=batch_size, test=False)
                valid_idx = None
                model = train_Net()
                criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor(1)
                criterion2 = torch.nn.MSELoss()
                criterion3 = TripletLoss()
                criterion4 = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
                StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
                model.num_epochs = main_num_epochs
                model = model.to(device)
                x_test = KinShipDataSetZL(relation, 'test', image_path, meta_data_path, train_idx, valid_idx, test_idx,
                                          fold, data_transforms['test'], aug=False)
                test_loader = DataLoader(x_test, batch_size=batch_size, num_workers=num_workers)
                model = train_model_no_val(model, dataloaders, criterion, criterion2, criterion3, criterion4, optimizer,
                                           fold, relation, test_loader, StepLR, scheduler=None, linear_momentum=False,
                                           early_stop=False)


if __name__ == '__main__':
    mainonlytest()
