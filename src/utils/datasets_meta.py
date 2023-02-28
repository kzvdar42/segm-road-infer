import numpy as np


_cityscapes_palette = np.zeros((256, 3), dtype=np.uint8)
_cityscapes_palette[:21] = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32], [255, 255, 255], [0, 0, 0]
]

_mapillary_palette = np.zeros((256, 3), dtype=np.uint8)
_mapillary_palette[:66] = [
    [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
    [180, 165, 180], [90, 120, 150], [102, 102, 156], [128, 64, 255],
    [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
    [230, 150, 140], [128, 64, 128], [110, 110, 110], [244, 35, 232],
    [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
    [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
    [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
    [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
    [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
    [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
    [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
    [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
    [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
    [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142], [0, 0, 90],
    [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110], [0, 0, 70],
    [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]
]

_my_data_palette = np.zeros((256, 3), dtype=np.uint8)
_my_data_palette[:26] = [
    [0, 0, 0], [0, 192, 0], [220, 20, 60], [255, 0, 0], [152, 251, 152],
    [107, 142, 35], [0, 170, 30], [70, 130, 180], [150, 100, 100], [70, 70, 70],
    [150, 120, 90], [90, 120, 150], [250, 0, 30], [180, 165, 180], [250, 170, 30],
    [128, 128, 128], [192, 192, 192], [220, 220, 0], [153, 153, 153], [230, 150, 140],
    [100, 128, 160], [70, 100, 150], [128, 64, 128], [96, 96, 96], [0, 0, 142], [120, 10, 10]
]

_ego_palette = np.zeros((256, 3), dtype=np.uint8)
_ego_palette[[0, 255]] = [[0, 0, 0], [0, 0, 255]]

datasets_meta = {
    'cityscapes': dict(palette=_cityscapes_palette, classes_file_path='datasets_meta/cityscapes-classes.txt'),
    'cityscapes+ego': dict(palette=_cityscapes_palette, classes_file_path='datasets_meta/cityscapes+ego-classes.txt'),
    'mapillary': dict(palette=_mapillary_palette, classes_file_path='datasets_meta/mapillary-classes.txt'),
    'my_data': dict(palette=_my_data_palette, classes_file_path='datasets_meta/my-data-classes.txt'),
    'ego_data': dict(palette=_ego_palette, classes_file_path=['background', 'ego_vehicle']),
}

def get_classes(dataset_type: str, add_ego_cls: bool):
    classes = []
    # If provided classes list, apply them
    if isinstance(dataset_type, list):
        classes = dataset_type
    elif dataset_type not in datasets_meta:
        raise ValueError(f'Unknown model dataset type! Got {dataset_type}')
    # Overvise load class list from file
    else:
        classes_file_path = datasets_meta[dataset_type]['classes_file_path']
        with open(classes_file_path) as in_file:
            classes = [c.strip() for c in in_file.read().split('\n') if c.strip()]
    
    if add_ego_cls:
        classes.append('ego_vehicle')

    cls_name_to_id = {c:c_id for c_id, c in enumerate(classes)}
    cls_id_to_name = {c_id:c for c, c_id in cls_name_to_id.items()}
    
    ego_class_ids = np.array([
        cls_name_to_id[cls_name] for cls_name in ['ego_vehicle', 'car_mount'] if cls_name in cls_name_to_id
    ])
    return classes, cls_name_to_id, cls_id_to_name, ego_class_ids

def get_palette(data_type):
    if data_type in datasets_meta:
        return datasets_meta[data_type]['palette']
    raise ValueError(f'Unknown data type ({data_type}), support only: {list(datasets_meta)}')
