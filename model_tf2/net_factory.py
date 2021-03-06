from model_tf2.fishnet import fish

def myfishnet(**kwargs):
    
    # Total params: 1,034,440
    # Trainable params: 1,021,224
    # Non-trainable params: 13,216
    
    net_cfg = {
        #  input size:   [64,  16,  8,   4  |  2,   2,   4,   8 | 16]
        # output size:   [16,  8,   4,   2  |  2,   4,   8,  16 |  8]
        #                  |    |    |   |     |    |    |    |    |
        'network_planes': [32, 64, 128, 256, 256, 256, 192, 128, 160],
        'num_res_blks': [2, 2, 2, 2, 1, 1, 1, 1, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 200,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def myfishnet2(**kwargs):

    # Total params: 1,797,960
    # Trainable params: 1,775,624
    # Non-trainable params: 22,336
    
    net_cfg = {
        #  input size:   [64,  16,  8,  4  |  2,   2,   4,   8 | 16]
        # output size:   [16,   8,  4,  2  |  2,   4,   8,  16 |  8]
        #                  |    |   |   |     |    |    |    |    | 
        'network_planes': [32, 64, 128, 256, 256, 256, 192, 128, 160],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 200,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)


def myfishnet3(**kwargs):
    
    # Total params: 667,584
    # Trainable params: 654,368
    # Non-trainable params: 13,216
    
    net_cfg = {
        #  input size:   [64,  16,  8  |  4   4,   8 | 16,  8]
        # output size:   [16,  8,   4  |  4   8,  16 |  8,  4]
        #                  |    |    |    |   |    |    |   |
        'network_planes': [32, 64, 128, 128, 128, 96, 128, 160],
        'num_res_blks': [2, 4, 8, 1, 1, 1, 2],
        'num_trans_blks': [1, 1, 1, 4],
        'num_cls': 200,
        'num_down_sample': 2,
        'num_up_sample': 2,
        'trans_map': (1,0,4,3),
    }
    cfg = {**net_cfg, **kwargs}
    return fish(**cfg)
