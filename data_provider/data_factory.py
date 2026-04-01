"""
数据工厂 - 为DEAP和SEED EEG数据集提供统一的数据加载接口
"""

from torch.utils.data import DataLoader
from data_provider.data_loader_eeg import Dataset_DEAP, Dataset_SEED, ChannelSelector

data_dict = {
    'DEAP': Dataset_DEAP,
    'SEED': Dataset_SEED,
}


def data_provider(args, flag):
    """
    创建数据集和数据加载器

    Args:
        args: 参数对象，需要包含 data, root_path, seq_len, batch_size 等
        flag: 'train' / 'val' / 'test'

    Returns:
        data_set: Dataset对象
        data_loader: DataLoader对象
    """
    if args.data not in data_dict:
        raise ValueError(f"不支持的数据集: {args.data}，可选: {list(data_dict.keys())}")

    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True

    batch_size = args.batch_size

    if args.data == 'DEAP':
        deap_args = {
            'root_path': args.root_path,
            'flag': flag,
            'seq_len': args.seq_len,
            'pred_len': getattr(args, 'pred_len', 0),
            'label_len': getattr(args, 'label_len', 0),
            'n_class': args.num_class,
            'classification_type': getattr(args, 'classification_type', 'valence'),
            'subject_list': getattr(args, 'subject_list', None),
            'overlap': getattr(args, 'overlap', 0),
            'normalize': getattr(args, 'normalize', True),
            'filter_freq': getattr(args, 'filter_freq', None),
            'sampling_rate': getattr(args, 'sampling_rate', 128),
            'channel_selection': getattr(args, 'channel_selection', 'auto'),
            'use_channel_selection': getattr(args, 'use_channel_selection', True),
        }
        data_set = Data(**deap_args)

        # 更新通道数（通道选择后可能变化）
        if hasattr(data_set, 'n_channels'):
            args.enc_in = data_set.n_channels
            args.dec_in = data_set.n_channels
            args.c_out = data_set.n_channels
            print(f"已更新模型输入通道数为: {data_set.n_channels}")

    elif args.data == 'SEED':
        seed_args = {
            'root_path': args.root_path,
            'flag': flag,
            'seq_len': args.seq_len,
            'pred_len': getattr(args, 'pred_len', 0),
            'label_len': getattr(args, 'label_len', 0),
            'n_class': args.num_class,
            'subject_list': getattr(args, 'subject_list', None),
            'overlap': getattr(args, 'overlap', 0),
            'normalize': getattr(args, 'normalize', True),
            'filter_freq': getattr(args, 'filter_freq', None),
            'sampling_rate': getattr(args, 'sampling_rate', 200),
        }
        data_set = Data(**seed_args)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
