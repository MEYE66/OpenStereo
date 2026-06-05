import random
import numpy as np

import torch

from stereo.datasets.carla_stereo_dataset import CarlaStereoDataset



'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3



class Dict(dict):
    """
      Example:
      m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
      """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


def create_input_tensor(batch):
    im_list, label_list, path_list, shapes_list, states_list = batch
    for i, lb in enumerate(label_list):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.from_numpy(np.stack(im_list, 0)), \
           torch.from_numpy(np.concatenate(label_list, 0)), path_list, shapes_list, \
           torch.from_numpy(np.stack(states_list, 0))


def get_noise(batch_size, z_type="uniform", z_dim=27):
    if z_type == 'normal':
        return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
    elif z_type == 'uniform':
        return np.random.uniform(0, 1, [batch_size, z_dim]).astype(np.float32)
    else:
        assert False, 'Unknown noise type: %s' % z_type


def get_initial_states(batch_size, num_state_dim, filters_number):
    states = np.zeros(shape=(batch_size, num_state_dim), dtype=np.float32)
    for k in range(batch_size):
        for i in range(len(filters_number)):
            # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
            # Used or not?
            # Initially nothing has been used
            states[k, -(i + 1)] = 0
    return states


class ReplayMemory:
    def __init__(self,
                 cfg,
                 load,
                 batch_size,
                 ):
        self.cfg = cfg
        self.dataset = CarlaStereoDataset()
        # The images with labels of #operations applied
        self.image_pool = []
        self.target_pool_size = cfg.replay_memory_size
        self.fake_output = None
        self.batch_size = batch_size
        if load:
            self.load()

    def load(self):
        self.fill_pool()

    def get_initial_states(self, batch_size):
        states = np.zeros(shape=(batch_size, self.cfg.num_state_dim), dtype=np.float32)
        for k in range(batch_size):
            for i in range(len(self.cfg.filters)):
                # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
                # Used or not?
                # Initially nothing has been used
                states[k, -(i + 1)] = 0
        return states

    def fill_pool(self):
        while len(self.image_pool) < self.target_pool_size:
            im_list, label_list, path_list, shapes_list = self.dataset.get_next_batch(self.batch_size)
            for i in range(len(im_list)):
                self.image_pool.append(Dict(
                    im=im_list[i],
                    label=label_list[i],
                    path=path_list[i],
                    shape=shapes_list[i],
                    state=self.get_initial_states(1)[0]))
        self.image_pool = self.image_pool[:self.target_pool_size]
        assert len(self.image_pool) == self.target_pool_size, '%d, %d' % (
            len(self.image_pool), self.target_pool_size)

    def get_next_RAW(self, batch_size):
        im_list, label_list, path_list, shapes_list = self.dataset.get_next_batch(batch_size)
        pool = []
        for i in range(len(im_list)):
            pool.append(Dict(
                im=im_list[i],
                label=label_list[i],
                path=path_list[i],
                shape=shapes_list[i],
                state=self.get_initial_states(1)[0]))
        return self.records_to_images_and_states(pool)

    def get_feed_dict_and_states(self, batch_size):
        images, labels, paths, shapes, states = self.get_next_fake_batch(batch_size)
        z = self.get_noise(batch_size)
        data = {
            "im": images,   # list
            "label": labels,  # list
            "path": paths,  # list
            "shape": shapes,  # list
            "state": states,  # list
            "z": z   # numpy
        }
        #
        return data

    # Not actually used.
    def get_noise(self, batch_size):
        if self.cfg.z_type == 'normal':
            return np.random.normal(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        elif self.cfg.z_type == 'uniform':
            return np.random.uniform(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        else:
            assert False, 'Unknown noise type: %s' % self.cfg.z_type

    # Note, we add finished images since the discriminator needs them for training.
    def replace_memory(self, new_images):
        random.shuffle(self.image_pool)
        # Insert only PART of new images
        for r in new_images:
            if r.state[STATE_STEP_DIM] < self.cfg.maximum_trajectory_length or random.random(
            ) < self.cfg.over_length_keep_prob:
                self.image_pool.append(r)
        # ... and add some brand-new RAW images
        self.fill_pool()
        random.shuffle(self.image_pool)

    # For supervised learning case, images should be [batch size, 2, channels, size, size]
    @staticmethod
    def records_to_images_and_states(batch):
        im_list = [x['im'] for x in batch]
        label_list = [x['label'] for x in batch]
        path_list = [x['path'] for x in batch]
        shapes_list = [x['shape'] for x in batch]
        states_list = [x['state'] for x in batch]
        return im_list, label_list, path_list, shapes_list, states_list
        # for i, lb in enumerate(label_list):
        #     lb[:, 0] = i  # add target image index for build_targets()
        # return np.stack(im_list, 0), np.concatenate(label_list, 0), path_list, shapes_list,
        # np.stack(states_list, axis=0)

    @staticmethod
    def images_and_states_to_records(images, labels, paths, shapes, states):
        assert len(images) == len(states)
        records = []
        for i in range(len(images)):
            records.append(Dict(
                im=images[i],
                label=labels[i],
                path=paths[i],
                shape=shapes[i],
                state=states[i]))
        return records

    def get_next_fake_batch(self, batch_size):
        # print('get_next')
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        batch = []
        while len(batch) < batch_size:
            if len(self.image_pool) == 0:
                self.fill_pool()
            record = self.image_pool[0]
            self.image_pool = self.image_pool[1:]
            if record.state[STATE_STOPPED_DIM] != 1:
                # We avoid adding any finished images here.
                batch.append(record)
        return self.records_to_images_and_states(batch)

    def debug(self):
        tot_trajectory = 0
        for r in self.image_pool:
            tot_trajectory += r.state[STATE_STEP_DIM]
        average_trajectory = 1.0 * tot_trajectory / len(self.image_pool)
        print('# Replay memory: size %d, avg. traj. %.2f' % (len(self.image_pool),
                                                             average_trajectory))
        print('#--------------------------------------------')


if __name__ == "__main__":
    import yaml
    from config import cfg

    cfg.replay_memory_size = 2

    train_path = "COCO/coco2017/val2017.txt"
    hyp = 'yolov3/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    memory = ReplayMemory(cfg, True, train_path, 512, 1, 32, False, hyp,
                          rect=False, prefix='train: ', limit=1)
    # test get batch data
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)

    feed_dict['path'][0] = "1"
    print(feed_dict['path'])
    # test update value
    memory.replace_memory(memory.images_and_states_to_records(feed_dict['im'], feed_dict['label'], feed_dict['path'],
                                                              feed_dict['shape'], feed_dict['state']))
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)