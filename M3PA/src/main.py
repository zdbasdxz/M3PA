

from models.GeoBERT_model.GeoBERT import GeoBERTConfig, GeoBertForSemanticTyping
from models.M3PA import M3PA
from dataloader.poi_image_loader import M3PA_Dataset
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
from transformers import  BertTokenizer






def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epoch
    max_token_len = args.max_token_len


    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    config = GeoBERTConfig(num_semantic_types=args.num_classes)

    print("Data_path:")
    print(args.load_data_path)


    train_dataset = M3PA_Dataset(
        poi_data_file_path=args.load_data_path,
        poi_streetimage_embedding_path=args.poi_streetimage_embedding_path,
        near_info_npz=args.near_info_npz,
        num_near=args.k,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        spatial_coordinate_fill=0,
        sep_between_neighbors=True,
        one_hot_shape=args.num_classes,
        mode='train')
    test_dataset = M3PA_Dataset(
        poi_data_file_path=args.load_data_path,
        poi_streetimage_embedding_path=args.poi_streetimage_embedding_path,
        near_info_npz=args.near_info_npz,
        num_near=args.k,
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        spatial_coordinate_fill=0,
        sep_between_neighbors=True,
        one_hot_shape=args.num_classes,
        mode='test')



    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=True, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)

    device = args.device

    bert = GeoBertForSemanticTyping(config)

    print("model_path:")
    print(args.load_model_path)
    bert.load_state_dict(torch.load(args.load_model_path), strict=False)

    loaded_weights = bert.state_dict()
    for name, param in bert.named_parameters():
        if name in loaded_weights:
            if not torch.equal(param, loaded_weights[name]):
                print(f"{name} 的权重没有被加载")

    print(args.num_classes)
    model = M3PA(
        geobert=bert,
        bert_input_dim=768,
        street_input_dim=512,
        hidden_dim=256,
        num_nearest=args.k,
        shape_input_phenomenon=512+args.num_classes,
        num_semantic_types=args.num_classes,
        )

    model.to(device)

    for param in model.geobert.parameters():
        param.requires_grad = False
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5E-5)

    print('start training...')
    print("save_model_path:")

    loss_list = []


    iter = 0

    for epoch in tqdm(range(epochs)):
        epoch_loss_list = []

        print("load_model_path:")
        print(args.load_model_path)
        print("load_data_path:")
        print(args.load_data_path)
        print("save_result_path:")
        print(args.base_result_outdir)
        model.train()
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            # initialize
            optim.zero_grad()
            # pull all tensor batches required for training
            #geobert
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_coordinate_list = batch['sent_coordinate'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)
            poi_lens = batch['poi_token_len'].to(device)

            #geoattention
            poi_streetimage_embedding = batch['poi_streetimage_embedding'].to(device)
            n_dist = batch['n_dist'].to(device)
            n_poi_streetimage_embedding = batch['n_poi_streetimage_embedding'].to(device)
            n_one_hot = batch['n_one_hot'].to(device)
            n_feature = torch.cat([n_poi_streetimage_embedding, n_one_hot], dim=2)

            labels = batch['poi_type'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                            sent_coordinate_list=sent_coordinate_list, labels=labels,
                            poi_len_list=poi_lens,
                            poi_street_embedding=poi_streetimage_embedding, near_feature=n_feature, n_dist=n_dist)


            loss = outputs[0]
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss': loss.item()})
            loss_list.append(loss.item())

            epoch_loss_list.append(loss.item())

            if iter %2000 == 0:
                plt_batch(args, loss_list)
            iter += 1

    test(test_loader, model, device, epoch, args.base_result_outdir, args.l)


def test(test_loader, model, device, epoch, base_result_outdir,l):
    model.eval()
    print("save_result_path:")
    print(base_result_outdir)
    if not os.path.exists(base_result_outdir):

        os.makedirs(base_result_outdir)
    header = ('l', 'epoch', 'Accuracy', 'F1-score', 'MRR')
    with open(base_result_outdir + 'result.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    with torch.no_grad():

        loop = tqdm(test_loader, leave=True)
        pred_list = []
        label_list = []
        for batch in loop:
            # geobert
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_coordinate_list = batch['sent_coordinate'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)
            poi_lens = batch['poi_token_len'].to(device)

            # geoattention
            poi_streetimage_embedding = batch['poi_streetimage_embedding'].to(device)
            n_dist = batch['n_dist'].to(device)
            n_poi_streetimage_embedding = batch['n_poi_streetimage_embedding'].to(device)
            n_one_hot = batch['n_one_hot'].to(device)
            n_feature = torch.cat([n_poi_streetimage_embedding, n_one_hot], dim=2)

            labels = batch['poi_type'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, sent_position_ids=sent_position_ids,
                            sent_coordinate_list=sent_coordinate_list, labels=labels,
                            poi_len_list=poi_lens,
                            poi_street_embedding=poi_streetimage_embedding, near_feature=n_feature, n_dist=n_dist)

            pred_list.extend(outputs[1].cpu().detach().numpy())
            label_list.extend(batch['poi_type'])
        y_testlabel = np.array(label_list)
        predictions_test = np.array(pred_list)
        predictions_test_dim = np.argmax(predictions_test, axis=1)
        accuracy_score_test = accuracy_score(y_testlabel, predictions_test_dim)
        f1_score_test = f1_score(y_testlabel, predictions_test_dim, average="macro")
        mrr_test = compute_mrr(y_testlabel, predictions_test)
        with open(base_result_outdir + 'result.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([l, epoch, accuracy_score_test, f1_score_test, mrr_test])

    return

def compute_mrr(true_labels, machine_preds):
    """Compute the MRR """
    rr_total = 0.0
    for i in range(len(true_labels)):
        ranklist = list(np.argsort(machine_preds[i])[::-1])  # 概率从大到小排序，返回index值
        rank = ranklist.index(true_labels[i]) + 1  # 获取真实值的rank
        rr_total = rr_total + 1.0 / rank
    mrr = rr_total / len(true_labels)
    return mrr
def plt_batch(args, lists):
    rcParams['figure.figsize'] = (8, 4)
    rcParams['figure.dpi'] = 100
    rcParams['font.size'] = 8
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.facecolor'] = '#ffffff'
    rcParams['lines.linewidth'] = 2.0
    rcParams['figure.figsize'] = (16, 6)
    plt.figure()
    # loss
    plt.plot(lists, 'g', label='train loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    if not os.path.exists(args.base_result_image_outdir):

        os.makedirs(args.base_result_image_outdir)
    plt.savefig(args.base_result_image_outdir + str(args.l) + ".png")
    plt.close()
class Config(object):
    PATH = os.path.dirname(os.path.realpath(__file__))
    PATH = PATH.replace('\\', '/')
    id_dataset = 'lixia'
    if id_dataset == "lixia":
        max_token_len_dcit = {
            15: 300,
        }
    if id_dataset == "haidian":
        max_token_len_dcit = {
            15: 360,
        }
    k = 15
    epoch = 12
    batch_size = 64

    max_token_len = max_token_len_dcit[k]
    r = 0
    print(PATH)
    bert_model_path = PATH[:-3] + "model_weight/bert-base-chinese"
    load_model_path = PATH[:-3] + "model_weight/" + id_dataset + "/geobert.pth"
    load_data_path = PATH[:-3] + "datasets/" + id_dataset + '/poi_data_neighbor_k_15.json'
    poi_streetimage_embedding_path = PATH[:-3] + "datasets/" + id_dataset + "/poi_streetscape.npy"
    near_info_npz = PATH[:-3] + "datasets/" + id_dataset + '/poi_street_data.npz'
    base_result_outdir = PATH[:-3] + "result/resulttxt/" + id_dataset + "/"
    base_result_image_outdir = PATH[:-3] + "result/image/" + id_dataset + "/"
    if id_dataset == "lixia":
        num_classes = 140

    elif id_dataset == "haidian":
        num_classes = 248
    lr=0.0001
    num_workers = 1
    device = torch.device('cuda:1')
if __name__ == '__main__':
    myconfig = Config()

    myconfig.l = 0
    training(myconfig)