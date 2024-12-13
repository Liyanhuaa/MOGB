
from pretrain import *


import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

class ModelManager:

    def __init__(self, args, data,pretrained_model=None):

        self.model = pretrained_model
        if self.model is None:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
            self.restore_model(args)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None







    def open_classify2(self, data,features):

        logits = euclidean_metric(features, self.centroids)  #算了输入特征 features 与已知类别的聚类中心（self.centroids）之间的欧氏距离
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        #这行代码将欧氏距离转换为概率分布，然后选择具有最大概率的类别。F.softmax 函数用于计算概率分布，max(dim=1) 用于选择最大概率的类别。probs 存储了最大概率，preds 存储了对应的类别索引
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        #这行代码计算了每个输入特征与其被分配到的类别的聚类中心之间的欧氏距离
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id
        #将距离大于等于 self.delta[preds] 的样本分配到未知类别（开放领域）的标签，通常用 data.unseen_token_id 表示未知类别

        return preds #这行代码返回分类结果，其中包括了对每个输入特征分配的类别标签

    def open_classify2(self,data, features,gb_centroids, gb_radii, gb_labels):
        all_centroids = []
        all_labels = []
        all_deltas = []

        # 构建质心、标签和半径的列表
        for label in gb_labels.keys():
            centroids = gb_centroids[label]
            deltas = gb_radii[label]
            for i in range(len(centroids)):
                all_centroids.append(centroids[i])
                all_labels.append(label)
                all_deltas.append(deltas[i])

        all_centroids = torch.stack(all_centroids)
        all_labels = torch.tensor(all_labels)
        all_deltas = torch.tensor(all_deltas)

        # 计算输入特征与所有质心之间的欧氏距离
        logits = euclidean_metric(features, all_centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)

        # 计算每个输入特征与其预测类别质心之间的欧氏距离
        euc_dis = torch.norm(features - all_centroids[preds], 2, 1).view(-1)

        # 初始化分类结果为未知类别
        final_preds = torch.tensor([data.unseen_token_id] * len(features))

        # 对每个样本进行分类
        for i in range(len(features)):
            if euc_dis[i] < all_deltas[preds[i]]:
                final_preds[i] = all_labels[preds[i]]

        return final_preds  # 返回分类结果，其中包括对每个输入特征分配的类别标签
    def euclidean_metric(self,a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return torch.sqrt(torch.pow(a - b, 2).sum(2))
    def open_classify(self, data, features, gb_centroids, gb_radii, gb_labels):
        # 计算输入特征与所有质心之间的欧氏距离
        logits = self.euclidean_metric(features, gb_centroids)
        _, preds = logits.min(dim=1)  # 获取最小距离的索引，对应最近的质心
        print("Features shape:", features.shape)
        print("Indexed centroids shape:", gb_centroids[preds].shape)

        # 计算每个输入特征与其预测类别质心之间的实际欧氏距离
        euc_dis = torch.norm(features - gb_centroids[preds], dim=1)

        # 初始化分类结果为未知类别
        final_preds = torch.tensor([data.unseen_token_id] * features.shape[0], device=features.device)

        # 对每个样本进行分类
        for i in range(features.shape[0]):
            if euc_dis[i] < gb_radii[preds[i]]:
                final_preds[i] = gb_labels[preds[i]]
            else:
                final_preds[i] = data.unseen_token_id  # 如果距离大于半径，标记为未知类别

        return final_preds


    def evaluation(self, args, data, gb_centroids, gb_radii, gb_labels,mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):  # 上下文管理器，用于暂时关闭梯度计算
                pooled_output, _ = self.model(input_ids, segment_ids,
                                              input_mask)  # 得到模型的输出特征表示 pooled_output。通常，这里的 _ 是一个占位符，用于接收不需要的输出
                preds = self.open_classify(data,pooled_output,gb_centroids, gb_radii, gb_labels)  # 将模型的输出特征表示输入到 self.open_classify 方法中，用于进行分类预测

                total_labels = torch.cat((total_labels, label_ids))  # 将当前批次的标签和预测结果追加到总的标签和预测张量中。
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = list([data.label_list[idx] for idx in y_pred])  # 将预测结果和真实标签转换为标签列表，以便进一步的分析和报告
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':
            data.y_true_no_noise = y_true
            data.y_pred_no_noise = y_pred

            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)
            print('Accuracy:', acc)
    def final_eval(self,args,data):
        y_true = np.concatenate((data.y_true_no_noise, data.y_true_noise), axis=0)
        y_pred = np.concatenate((data.y_pred_no_noise, data.y_pred_noise), axis=0)

        cm = confusion_matrix(y_true, y_pred)
        results = F_measure(cm)
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        results['Accuracy'] = acc
        self.test_results = results
        self.save_results(args)
        print('Accuracy:',acc)



    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num



    def restore_model(self, args): #从磁盘上加载预训练模型的权重并将它们应用到模型实例中
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME) #构建了要加载的预训练模型权重文件的路径,使用了 os.path.join 函数将 args.pretrain_dir 和 WEIGHTS_NAME 连接起来，以获得完整的文件路径
        self.model.load_state_dict(torch.load(output_model_file))
        # PyTorch 的 torch.load 函数从指定路径 output_model_file 加载模型权重。加载后的权重被应用到 self.model 中，即当前类实例中的模型对象
    
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        



        file_name = 'results.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)  #将要保存的结果数据（可能是一组数值或列表）添加到 ori 列表
            df1 = pd.DataFrame(ori,columns = keys)  #创建一个 Pandas 数据帧（DataFrame），将 ori 列表中的数据存储在数据帧中，并指定列名为 keys 中定义的列名
            df1.to_csv(results_path,index=False)  #将数据帧 df1 保存为 CSV 文件，指定 index=False 参数以避免保存索引列
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            #df1 = df1.append(new,ignore_index=True)
            df1 = pd.concat([df1, new], ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path) #代码重新读取整个 CSV 文件，将其加载到数据帧 data_diagram 中
        
        print('test_results', data_diagram)



##画图加到最后

    def plot_centroids_and_subcentroids(self,data,centroids, delta, true_subcentroids, radii_matrix):
        # 假设数据已经转换为CPU
        centroids = centroids.cpu().numpy()  # 假设已经是numpy数组
        delta = delta.detach().cpu().numpy()  # 假设已经转换为numpy数组
        radii_matrix = radii_matrix.cpu().numpy()  # 假设已经是numpy数组
        true_subcentroids = true_subcentroids.cpu().numpy()  # 假设已经转换为numpy数组

        # 使用TSNE进行降维，这里我们需要将子质心重新整理为一个二维数组
        sub_centroids_reshaped = true_subcentroids.reshape(-1, 768)  # 变形为60x768
        all_points = np.vstack([centroids, sub_centroids_reshaped])
        transformed_points = TSNE(n_components=2).fit_transform(all_points)

        # 分离降维后的质心和子质心
        transformed_centroids = transformed_points[:data.num_labels]
        transformed_sub_centroids = transformed_points[data.num_labels:]

        # 为15个类生成不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, 15))

        # 绘图
        fig, ax = plt.subplots()
        for i in range(data.num_labels):
            # 主质心
            ax.scatter(transformed_centroids[i, 0], transformed_centroids[i, 1], color=colors[i],
                       label=f'Class {i + 1}', edgecolor='black')
            circle = plt.Circle((transformed_centroids[i, 0], transformed_centroids[i, 1]), delta[i], color=colors[i],
                                fill=False, linewidth=2)
            ax.add_artist(circle)

            # 子质心
            for j in range(4):
                idx = i * 4 + j  # 为子质心计算正确的索引
                ax.scatter(transformed_sub_centroids[idx, 0], transformed_sub_centroids[idx, 1], color=colors[i],
                           alpha=0.5)
                circle = plt.Circle((transformed_sub_centroids[idx, 0], transformed_sub_centroids[idx, 1]),
                                    radii_matrix[i, j], color=colors[i], fill=False, linestyle='--', alpha=0.5)
                ax.add_artist(circle)

        # 添加图例和调整布局
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    
    
