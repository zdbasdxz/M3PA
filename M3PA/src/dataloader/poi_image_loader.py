import numpy as np
import torch
from torch.utils.data import Dataset
import json







class GeoBERTDataset(Dataset):
    def __init__(self, tokenizer, max_token_len, sep_between_neighbors=True):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.sep_between_neighbors = sep_between_neighbors

    def parse_spatial_context(self, poi_name, poi_coordinate, neighbor_name_list, neighbor_coordinate_list,
                              spatial_coordinate_fill):

        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        max_token_len = self.max_token_len

        poi_name_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(poi_name))
        poi_token_len = len(poi_name_tokens)

        pivot_lng = poi_coordinate[0]
        pivot_lat = poi_coordinate[1]

        # prepare entity mask
        entity_mask_arr = []
        # random number for masking
        rand_entity = np.random.uniform(
            size=len(neighbor_name_list) + 1)


        # 15%
        if rand_entity[0] < 0.15:
            entity_mask_arr.extend([True] * poi_token_len)
        else:
            entity_mask_arr.extend([False] * poi_token_len)

        # process neighbors
        neighbor_token_list = []
        neighbor_lng_list = []
        neighbor_lat_list = []
        token_len_all = poi_token_len + 1
        # add separator between poi and neighbor tokens

        if self.sep_between_neighbors:
            neighbor_lng_list.append(spatial_coordinate_fill)
            neighbor_lat_list.append(spatial_coordinate_fill)
            neighbor_token_list.append(sep_token_id)
            token_len_all = token_len_all + 1  # 用来限制长度

        for neighbor_name, neighbor_coordinate, rnd in zip(neighbor_name_list, neighbor_coordinate_list, rand_entity[1:]):

            if not neighbor_name[0].isalpha():

                continue

            neighbor_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neighbor_name))
            neighbor_token_len = len(neighbor_token)
            if token_len_all + neighbor_token_len + 1 > max_token_len:
                break
            else:
                token_len_all = token_len_all + neighbor_token_len


            neighbor_lng_list.extend([neighbor_coordinate['coordinates'][0]] * neighbor_token_len)
            neighbor_lat_list.extend([neighbor_coordinate['coordinates'][1]] * neighbor_token_len)
            neighbor_token_list.extend(neighbor_token)

            if self.sep_between_neighbors:
                neighbor_lng_list.append(spatial_coordinate_fill)
                neighbor_lat_list.append(spatial_coordinate_fill)
                neighbor_token_list.append(sep_token_id)

                entity_mask_arr.extend([False])
                token_len_all = token_len_all + 1

            if rnd < 0.15:
                # True: mask out, False: Keey original token
                entity_mask_arr.extend([True] * neighbor_token_len)
            else:
                entity_mask_arr.extend([False] * neighbor_token_len)

        pseudo_sentence = poi_name_tokens + neighbor_token_list
        sentence_lng_list = [pivot_lng] * poi_token_len + neighbor_lng_list
        sentence_lat_list = [pivot_lat] * poi_token_len + neighbor_lat_list

        # including cls and sep
        sent_len = len(pseudo_sentence)

        max_token_len_middle = max_token_len - 2

        # padding and truncation
        if sent_len > max_token_len_middle:
            pseudo_sentence = [cls_token_id] + pseudo_sentence[:max_token_len_middle] + [sep_token_id]
            sentence_lat_list = [spatial_coordinate_fill] + sentence_lat_list[:max_token_len_middle] + [spatial_coordinate_fill]
            sentence_lng_list = [spatial_coordinate_fill] + sentence_lng_list[:max_token_len_middle] + [spatial_coordinate_fill]
            attention_mask = [False] + [1] * max_token_len_middle + [False]  # attention
        else:
            pad_len = max_token_len_middle - sent_len
            assert pad_len >= 0

            pseudo_sentence = [cls_token_id] + pseudo_sentence + [sep_token_id] + [pad_token_id] * pad_len
            sentence_lat_list = [spatial_coordinate_fill] + sentence_lat_list + [spatial_coordinate_fill] + [spatial_coordinate_fill] * pad_len
            sentence_lng_list = [spatial_coordinate_fill] + sentence_lng_list + [spatial_coordinate_fill] + [spatial_coordinate_fill] * pad_len
            attention_mask = [False] + [1] * sent_len + [0] * pad_len + [False]



        # mask entity in the pseudo sentence
        entity_mask_indices = np.where(entity_mask_arr)[0]
        masked_entity_input = [mask_token_id if i in entity_mask_indices else pseudo_sentence[i] for i in
                               range(0, max_token_len)]

        # mask token in the pseudo sentence
        rand_token = np.random.uniform(size=len(pseudo_sentence))
        # do not mask out cls and sep token
        token_mask_arr = (rand_token < 0.15) & (np.array(pseudo_sentence) != cls_token_id) & (
                    np.array(pseudo_sentence) != sep_token_id) & (np.array(pseudo_sentence) != pad_token_id)
        token_mask_indices = np.where(token_mask_arr)[0]

        masked_token_input = [mask_token_id if i in token_mask_indices else pseudo_sentence[i] for i in
                              range(0, max_token_len)]

        # yield masked_token with 50% prob, masked_entity with 50% prob
        if np.random.rand() > 0.5:
            masked_input = torch.tensor(masked_entity_input)
        else:
            masked_input = torch.tensor(masked_token_input)
        sentence_lat_list = np.expand_dims(sentence_lat_list, axis=0)
        sentence_lng_list = np.expand_dims(sentence_lng_list, axis=0)
        train_data = {}
        train_data['poi_name'] = poi_name
        train_data['poi_token_len'] = poi_token_len
        train_data['masked_input'] = masked_input
        train_data['sent_position_ids'] = torch.tensor(np.arange(0, len(pseudo_sentence)))
        train_data['attention_mask'] = torch.tensor(attention_mask)
        train_data['sent_coordinate'] = torch.tensor(np.concatenate((sentence_lng_list, sentence_lat_list), axis=0).transpose(1, 0)).to(torch.float32)
        train_data['pseudo_sentence'] = torch.tensor(pseudo_sentence)

        return train_data

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError






class M3PA_Dataset(GeoBERTDataset):
    def __init__(self, poi_data_file_path,
                 poi_streetimage_embedding_path,
                 near_info_npz,
                 num_near,
                 tokenizer=None,
                 max_token_len=512,
                 spatial_coordinate_fill=0,
                 sep_between_neighbors=True,
                 mode=None,
                 num_neighbor_limit=None,
                 random_remove_neighbor=0.,
                 one_hot_shape=140,
                 type_key_str='class'):


        self.tokenizer = tokenizer

        self.type_key_str = type_key_str  # key name of the class type in the input data dictionary

        self.max_token_len = max_token_len
        self.spatial_coordinate_fill = spatial_coordinate_fill
        self.sep_between_neighbors = sep_between_neighbors
        self.num_neighbor_limit = num_neighbor_limit

        self.random_remove_neighbor = random_remove_neighbor
        self.read_file(poi_data_file_path, poi_streetimage_embedding_path, near_info_npz, num_near,
                       mode, one_hot_shape)

        super(M3PA_Dataset, self).__init__(self.tokenizer, max_token_len, sep_between_neighbors)

    def read_file(self, poi_data_file_path, poi_streetimage_embedding_path, near_info_npz, num_near, mode,one_hot_shape):
        with open(poi_data_file_path, 'r') as f:
            poi_data = f.readlines()

        poi_streetimage_embedding = np.load(poi_streetimage_embedding_path)
        near_info = np.load(near_info_npz, allow_pickle=True)
        dist_geo_withouttest = near_info['dist_geo_withouttest'][:, :num_near]
        idx_geo_withouttest = near_info['idx_geo_withouttest'][:, :num_near]
        one_hot = near_info['onehot_ary'][:, -one_hot_shape:]

        self.all_poi_streetimage_embedding = poi_streetimage_embedding
        train_index = int(len(poi_data) * 0.8)
        if mode == 'train':
            poi_streetimage_embedding = poi_streetimage_embedding[0:train_index]

            dist_geo_withouttest = dist_geo_withouttest[0:train_index, :]
            idx_geo_withouttest = idx_geo_withouttest[0:train_index, :]
            labels = np.argmax(near_info['y_train'], axis=1)
            poi_data = poi_data[0:train_index]
        elif mode == 'test':
            poi_streetimage_embedding = poi_streetimage_embedding[train_index:]

            dist_geo_withouttest = dist_geo_withouttest[train_index:, :]
            idx_geo_withouttest = idx_geo_withouttest[train_index:, :]
            labels = np.argmax(near_info['y_test'], axis=1)
            poi_data = poi_data[train_index:]
        elif mode is None:
            pass
        else:
            raise NotImplementedError

        self.len_data = len(poi_data)  # updated data length
        self.poi_streetimage_embedding = poi_streetimage_embedding

        self.dist_geo_withouttest = dist_geo_withouttest
        self.idx_geo_withouttest = idx_geo_withouttest
        self.one_hot = one_hot
        self.labels =labels
        self.poi_data = poi_data

    def load_data(self, index):

        spatial_coordinate_fill = self.spatial_coordinate_fill
        poi_line = self.poi_data[index]

        poi_line_data_dict = json.loads(poi_line)

        # process pivot
        poi_name = poi_line_data_dict['info']['name']
        poi_coordinate = poi_line_data_dict['info']['coordinates']

        neighbor_info = poi_line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_coordinate_list = neighbor_info['coordinates_list']

        if self.random_remove_neighbor != 0:
            num_neighbors = len(neighbor_name_list)
            rand_neighbor = np.random.uniform(size=num_neighbors)

            neighbor_keep_arr = (rand_neighbor >= self.random_remove_neighbor)  # select the neighbors to be removed
            neighbor_keep_arr = np.where(neighbor_keep_arr)[0]

            new_neighbor_name_list, new_neighbor_coordinate_list = [], []
            for i in range(0, num_neighbors):
                if i in neighbor_keep_arr:
                    new_neighbor_name_list.append(neighbor_name_list[i])
                    new_neighbor_coordinate_list.append(neighbor_coordinate_list[i])

            neighbor_name_list = new_neighbor_name_list
            neighbor_coordinate_list = new_neighbor_coordinate_list

        if self.num_neighbor_limit is not None:
            neighbor_name_list = neighbor_name_list[0:self.num_neighbor_limit]
            neighbor_coordinate_list = neighbor_coordinate_list[0:self.num_neighbor_limit]

        train_data = self.parse_spatial_context(poi_name, poi_coordinate, neighbor_name_list, neighbor_coordinate_list,
                                                spatial_coordinate_fill)


        poi_type = poi_line_data_dict['info'][self.type_key_str]
        train_data['poi_type'] = torch.tensor(poi_type)

        train_data['poi_streetimage_embedding'] = torch.tensor(self.poi_streetimage_embedding[index])
        train_data['n_dist'] = torch.tensor(self.dist_geo_withouttest[index])

        train_data['n_poi_streetimage_embedding'] = torch.tensor(self.all_poi_streetimage_embedding[self.idx_geo_withouttest[index]])
        train_data['n_one_hot'] = torch.tensor(self.one_hot[self.idx_geo_withouttest[index]])

        return train_data






    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)