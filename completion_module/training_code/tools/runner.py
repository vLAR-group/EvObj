import time

import torch
import torch.nn as nn

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from training_code.tools import builder
from training_code.tools.average_meter import AverageMeter
from utils import dist_utils, misc
from utils.logger import get_logger, print_log
from utils.metrics import Metrics


SUPPORTED_DATASETS = {'ShapeNet6clsDataset'}


def _move_batch_to_cuda(config_dataset_name, partial, gt):
    if config_dataset_name not in SUPPORTED_DATASETS:
        raise NotImplementedError(f'Train/Test phase does not support {config_dataset_name}')
    return partial.cuda(non_blocking=True), gt.cuda(non_blocking=True)


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    (train_sampler, train_dataloader), (_, val_dataloader) = (
        builder.dataset_builder(args, config.dataset.train),
        builder.dataset_builder(args, config.dataset.val),
    )

    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    start_epoch = 0
    best_metrics = None
    metrics = None

    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[args.local_rank % max(torch.cuda.device_count(), 1)],
            find_unused_parameters=True,
        )
        print_log('Using Distributed DataParallel ...', logger=logger)
    else:
        if args.use_gpu:
            base_model = nn.DataParallel(base_model).cuda()
            print_log('Using DataParallel ...', logger=logger)
        else:
            print_log('Using CPU mode ...', logger=logger)

    optimizer = builder.build_optimizer(base_model, config)
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch - 1)

    chamfer_l1 = ChamferDistanceL1()
    chamfer_l2 = ChamferDistanceL2()

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0
        n_batches = len(train_dataloader)
        dataset_name = config.dataset.train._base_.NAME

        for idx, (_, _, partial, gt) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)

            if args.use_gpu:
                partial, gt = _move_batch_to_cuda(dataset_name, partial, gt)

            num_iter += 1
            ret = base_model(partial)
            model_ref = base_model.module if hasattr(base_model, 'module') else base_model
            sparse_loss, dense_loss = model_ref.get_loss(ret, gt, epoch)

            total_loss = sparse_loss + dense_loss
            total_loss.backward()

            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)

            losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            if args.distributed and args.use_gpu:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log(
                    '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f'
                    % (
                        epoch,
                        config.max_epoch,
                        idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        ['%.4f' % l for l in losses.val()],
                        optimizer.param_groups[0]['lr'],
                    ),
                    logger=logger,
                )

            if config.scheduler.type == 'GradualWarmup' and n_itr < config.scheduler.kwargs_2.total_epoch:
                scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()

        epoch_end_time = time.time()
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)

        print_log(
            '[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s'
            % (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]),
            logger=logger,
        )

        if epoch % args.val_freq == 0:
            metrics = validate(base_model, val_dataloader, epoch, chamfer_l1, chamfer_l2, val_writer, args, config, logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger=logger)

    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, val_dataloader, epoch, chamfer_l1, chamfer_l2, val_writer, args, config, logger=None):
    print_log(f'[VALIDATION] Start validating epoch {epoch}', logger=logger)
    base_model.eval()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = {}
    n_samples = len(val_dataloader)
    interval = max(n_samples // 10, 1)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, partial, gt) in enumerate(val_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else int(taxonomy_ids[0])
            model_id = model_ids[0]
            dataset_name = config.dataset.val._base_.NAME

            if args.use_gpu:
                partial, gt = _move_batch_to_cuda(dataset_name, partial, gt)

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss_l1 = chamfer_l1(coarse_points, gt)
            sparse_loss_l2 = chamfer_l2(coarse_points, gt)
            dense_loss_l1 = chamfer_l1(dense_points, gt)
            dense_loss_l2 = chamfer_l2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([
                sparse_loss_l1.item() * 1000,
                sparse_loss_l2.item() * 1000,
                dense_loss_l1.item() * 1000,
                dense_loss_l2.item() * 1000,
            ])

            metrics_vals = Metrics.get(dense_points, gt)
            if args.distributed:
                metrics_vals = [dist_utils.reduce_tensor(m, args).item() for m in metrics_vals]
            else:
                metrics_vals = [m.item() for m in metrics_vals]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(metrics_vals)

            if val_writer is not None and idx % 200 == 0:
                input_pc = misc.get_ptcloud_img(partial.squeeze().detach().cpu().numpy())
                val_writer.add_image(f'Model{idx:02d}/Input', input_pc, epoch, dataformats='HWC')

                sparse_img = misc.get_ptcloud_img(coarse_points.squeeze().detach().cpu().numpy())
                val_writer.add_image(f'Model{idx:02d}/Sparse', sparse_img, epoch, dataformats='HWC')

                dense_img = misc.get_ptcloud_img(dense_points.squeeze().detach().cpu().numpy())
                val_writer.add_image(f'Model{idx:02d}/Dense', dense_img, epoch, dataformats='HWC')

                gt_img = misc.get_ptcloud_img(gt.squeeze().detach().cpu().numpy())
                val_writer.add_image(f'Model{idx:02d}/DenseGT', gt_img, epoch, dataformats='HWC')

            if (idx + 1) % interval == 0:
                print_log(
                    'Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s'
                    % (
                        idx + 1,
                        n_samples,
                        taxonomy_id,
                        model_id,
                        ['%.4f' % l for l in test_losses.val()],
                        ['%.4f' % m for m in metrics_vals],
                    ),
                    logger=logger,
                )

        for v in category_metrics.values():
            test_metrics.update(v.avg())

        print_log('[Validation] EPOCH: %d Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed and args.use_gpu:
            torch.cuda.synchronize()

    print_log('============================ TEST RESULTS ============================', logger=logger)
    head = 'Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items)
    print_log(head, logger=logger)

    for taxonomy_id, meter in category_metrics.items():
        row = f'{taxonomy_id}\t{meter.count(0)}\t' + '\t'.join(['%.3f' % v for v in meter.avg()])
        print_log(row, logger=logger)

    overall = 'Overall\t\t' + '\t'.join(['%.3f' % v for v in test_metrics.avg()])
    print_log(overall, logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar(f'Metric/{metric}', test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ...', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model = nn.DataParallel(base_model).cuda()

    chamfer_l1 = ChamferDistanceL1()
    chamfer_l2 = ChamferDistanceL2()

    validate(base_model, test_dataloader, epoch=0, chamfer_l1=chamfer_l1, chamfer_l2=chamfer_l2,
             val_writer=None, args=args, config=config, logger=logger)
