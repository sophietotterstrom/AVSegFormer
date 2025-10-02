import torch
import torch.nn
import os
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
import csv

def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights, map_location='cuda:0'))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # write csv file with predictions
    csv_path = os.path.join(args.save_dir, dir_name)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path, exist_ok=True)
    write_file = os.path.join(csv_path, "output.csv")
    with open(write_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "video_name", "category", "miou", "F_score"])  # header
    

        # Test
        with torch.no_grad():
            for n_iter, batch_data in enumerate(test_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
                imgs, audio, mask, category_list, video_name_list = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2],
                                audio.shape[3], audio.shape[4])

                output, _ = model(audio, imgs)
                if args.save_pred_mask:
                    mask_save_path = os.path.join(
                        args.save_dir, dir_name, 'pred_masks')
                    save_mask(output.squeeze(1), mask_save_path,
                            category_list, video_name_list)
                    logger.info(f"INSIDE IF, save to {mask_save_path}, {video_name_list}")

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})
                F_score = Eval_Fmeasure(output.squeeze(1), mask)
                avg_meter_F.add({'F_score': F_score})
                logger.info(f'n_iter: {n_iter}, iou: {miou}, F_score: {F_score}')

                writer.writerow([n_iter, video_name_list[0], category_list[0], miou.item(), F_score])

                

            miou = (avg_meter_miou.pop('miou'))
            F_score = (avg_meter_F.pop('F_score'))
            logger.info(f'test miou: {miou.item}')
            logger.info(f'test F_score: {F_score}')
            logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('weights', type=str, help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')

    args = parser.parse_args()
    main()
