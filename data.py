from config import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

'''
assume classes across domains are the same.
[0 1 .......................................... N - 1]
|----common classes --||----source private classes --|
'''

source_classes = [i for i in range(args.data.dataset.n_total)]
if isinstance(args.data.dataset.n_share,list):
    target_classes = [i for i in range(*args.data.dataset.n_share)]
else:
    target_classes = [i for i in range(args.data.dataset.n_share)]

train_transform = Compose([
    # Grayscale(3),
    Resize(36),
    RandomCrop(32),
    RandomHorizontalFlip(),
    ToTensor()
])

test_transform = Compose([
    # Grayscale(3),
    Resize(36),
    CenterCrop(32),
    ToTensor()
])

from torch.utils.data import Subset

if args.data.dataset.name == 'digital':
    source_train_ds = source_file('~/', download=True, transform=train_transform, train=True)
    source_train_ds = Subset(source_train_ds,
                             [idx for idx, data in enumerate(source_train_ds) if data[1] in source_classes])
    source_test_ds = source_file('~/', download=True, transform=test_transform, train=False)
    source_test_ds = Subset(source_test_ds,
                            [idx for idx, data in enumerate(source_test_ds) if data[1] in source_classes])
    target_train_ds = target_file('~/', download=True, transform=train_transform, train=True)
    target_train_ds = Subset(target_train_ds,
                             [idx for idx, data in enumerate(target_train_ds) if data[1] in target_classes])
    target_test_ds = target_file('~/', download=True, transform=test_transform, train=False)
    target_test_ds = Subset(target_test_ds,
                            [idx for idx, data in enumerate(target_test_ds) if data[1] in target_classes])
    ans = [data for data in source_train_ds]
    _, source_train_ds.labels = zip(*ans)
else:
    source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                      transform=train_transform, filter=(lambda x: x in source_classes))
    source_test_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                     transform=test_transform, filter=(lambda x: x in source_classes))
    target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                      transform=train_transform, filter=(lambda x: x in target_classes))
    target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                     transform=test_transform, filter=(lambda x: x in target_classes))


classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x: 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
