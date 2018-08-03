from __future__ import division

import argparse
import functools
import numpy as np
import six

import chainer
from chainer.dataset.convert import _concat_arrays
from chainer.dataset.convert import to_device
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger
from chainercv.chainer_experimental.datasets.sliceable \
    import ConcatenatedDataset
from chainercv.chainer_experimental.datasets.sliceable \
    import TransformDataset
from chainercv.datasets import coco_instance_segmentation_label_names
from chainercv.datasets import COCOInstanceSegmentationDataset
from chainercv.extensions import InstanceSegmentationCOCOEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv import transforms
from chainercv.utils.mask.mask_to_bbox import mask_to_bbox
import chainermn

from psroi_align.links.model import FCISPSROIAlignResNet101
from psroi_align.links.model import FCISTrainChain


def concat_examples(batch, device=None, padding=None,
                    indices_concat=None, indices_to_device=None):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    elem_size = len(first_elem)
    if indices_concat is None:
        indices_concat = range(elem_size)
    if indices_to_device is None:
        indices_to_device = range(elem_size)

    result = []
    if not isinstance(padding, tuple):
        padding = [padding] * elem_size

    for i in six.moves.range(elem_size):
        res = [example[i] for example in batch]
        if i in indices_concat:
            res = _concat_arrays(res, padding[i])
        if i in indices_to_device:
            if i in indices_concat:
                res = to_device(device, res)
            else:
                res = [to_device(device, r) for r in res]
        result.append(res)

    return tuple(result)


class Transform(object):

    def __init__(self, fcis):
        self.fcis = fcis

    def __call__(self, in_data):
        img, mask, label = in_data
        bbox = mask_to_bbox(mask)
        _, orig_H, orig_W = img.shape
        img = self.fcis.prepare(img)
        _, H, W = img.shape
        scale = H / orig_H
        mask = transforms.resize(mask.astype(np.float32), (H, W))
        bbox = transforms.resize_bbox(bbox, (orig_H, orig_W), (H, W))

        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        mask = transforms.flip(mask, x_flip=params['x_flip'])
        bbox = transforms.flip_bbox(bbox, (H, W), x_flip=params['x_flip'])
        return img, mask, label, bbox, scale


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: FCIS')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument(
        '--lr', '-l', type=float, default=0.0005,
        help='Default value is for 1 GPU.\n'
             'The learning rate will be multiplied by the number of gpu')
    args = parser.parse_args()

    # chainermn
    comm = chainermn.create_communicator()
    device = comm.intra_rank

    np.random.seed(args.seed)

    # model
    proposal_creator_params = {
        'nms_thresh': 0.7,
        'n_train_pre_nms': 12000,
        'n_train_post_nms': 2000,
        'n_test_pre_nms': 6000,
        'n_test_post_nms': 1000,
        'force_cpu_nms': False,
        'min_size': 0
    }

    fcis = FCISPSROIAlignResNet101(
        n_fg_class=len(coco_instance_segmentation_label_names),
        min_size=800, max_size=1333,
        anchor_scales=(2, 4, 8, 16, 32),
        pretrained_model='imagenet', iter2=False,
        proposal_creator_params=proposal_creator_params)
    fcis.use_preset('coco_evaluate')
    model = FCISTrainChain(fcis)

    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    # dataset
    train_dataset = TransformDataset(
        ConcatenatedDataset(
            COCOInstanceSegmentationDataset(split='train'),
            COCOInstanceSegmentationDataset(split='valminusminival')),
        ('img', 'mask', 'label', 'bbox', 'scale'),
        Transform(model.fcis))
    test_dataset = COCOInstanceSegmentationDataset(
        split='minival', use_crowded=True,
        return_crowded=True, return_area=True)
    if comm.rank == 0:
        indices = np.arange(len(train_dataset))
    else:
        indices = None
    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train_dataset = train_dataset.slice[indices]
    train_iter = chainer.iterators.SerialIterator(train_dataset, batch_size=1)

    if comm.rank == 0:
        test_iter = chainer.iterators.SerialIterator(
            test_dataset, batch_size=1, repeat=False, shuffle=False)

    # optimizer
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=args.lr * comm.size, momentum=0.9),
        comm)
    optimizer.setup(model)

    model.fcis.head.conv1.W.update_rule.add_hook(GradientScaling(3.0))
    model.fcis.head.conv1.b.update_rule.add_hook(GradientScaling(3.0))
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    for param in model.params():
        if param.name in ['beta', 'gamma']:
            param.update_rule.enabled = False
    model.fcis.extractor.conv1.disable_update()
    model.fcis.extractor.res2.disable_update()

    converter = functools.partial(
        concat_examples, padding=0,
        # img, masks, labels, bboxes, scales
        indices_concat=[0, 1, 2, 4],  # img, masks, labels, _, scales
        indices_to_device=[0],        # img
    )

    updater = chainer.training.updater.StandardUpdater(
        train_iter, optimizer, converter=converter,
        device=device)

    trainer = chainer.training.Trainer(
        updater, (18, 'epoch'), out=args.out)

    # lr scheduler
    trainer.extend(
        chainer.training.extensions.ExponentialShift(
            'lr', 0.1, init=args.lr * comm.size),
        trigger=ManualScheduleTrigger([12, 15], 'epoch'))

    if comm.rank == 0:
        # interval
        log_interval = 100, 'iteration'
        plot_interval = 3000, 'iteration'
        print_interval = 10, 'iteration'

        # training extensions
        model_name = model.fcis.__class__.__name__
        trainer.extend(
            chainer.training.extensions.snapshot_object(
                model.fcis,
                savefun=chainer.serializers.save_npz,
                filename='%s_model_iter_{.updater.iteration}.npz'
                         % model_name),
            trigger=(1, 'epoch'))
        trainer.extend(
            extensions.observe_lr(),
            trigger=log_interval)
        trainer.extend(
            extensions.LogReport(log_name='log.json', trigger=log_interval))
        report_items = [
            'iteration', 'epoch', 'elapsed_time', 'lr',
            'main/loss',
            'main/rpn_loc_loss',
            'main/rpn_cls_loss',
            'main/roi_loc_loss',
            'main/roi_cls_loss',
            'main/roi_mask_loss',
            'validation/main/map/iou=0.50:0.95/area=all/max_dets=100',
        ]

        trainer.extend(
            extensions.PrintReport(report_items), trigger=print_interval)
        trainer.extend(
            extensions.ProgressBar(update_interval=10))

        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(
                    ['main/loss'],
                    file_name='loss.png', trigger=plot_interval),
                trigger=plot_interval)

        trainer.extend(
            InstanceSegmentationCOCOEvaluator(
                test_iter, model.fcis,
                label_names=coco_instance_segmentation_label_names),
            trigger=ManualScheduleTrigger(
                [len(train_dataset) * 12,
                 len(train_dataset) * 15], 'iteration'))

        trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
