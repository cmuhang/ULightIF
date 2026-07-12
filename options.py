import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Run mode')

    # Data paths
    parser.add_argument('--vi_train_path', type=str, default='/root/gpufree-data/datasets/train/vi', help='Visible image training path')
    parser.add_argument('--ir_train_path', type=str, default='/root/gpufree-data/datasets/train/ir', help='Infrared image training path')
    parser.add_argument('--test_root', type=str, default='/root/gpufree-data/datasets/test/msrs/', help='Root folder for paired test images')

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs per stage')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--crop_size', type=int, default=96, help='Crop size for training')

    # Optimizer settings
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--optim_step', type=int, default=50, help='Scheduler step size')
    parser.add_argument('--optim_gamma', type=float, default=0.5, help='Scheduler gamma')


    # 权重以及测试输出路径
    parser.add_argument('--save_dir', type=str, default='runs/train/checkpoints', help='Save directory')
    parser.add_argument('--output_dir', type=str, default='runs/test/outputs', help='Test output directory')
    parser.add_argument('--stage1_ckpt', type=str, default='runs/train/checkpoints/20260712_084452/IE/last.ckpt', help='Optional stage 1 checkpoint path')
    parser.add_argument('--stage2_ckpt', type=str, default='runs/train/checkpoints/20260712_084452/IF/last.ckpt', help='Optional stage 2 checkpoint path')
    parser.add_argument('--stage3_ckpt', type=str, default='runs/train/checkpoints/20260712_084452/ID/last.ckpt', help='Optional stage 3 checkpoint path')




    # Inference controls
    parser.add_argument('--check_fold', default=True, help='Check folded vs unfodled model outputs')
    parser.add_argument('--calc_flops', default=True, help='Calculate FLOPs before inference')
    parser.add_argument('--enable_timing', default=True, help='Measure inference time')


    # Reproducibility
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    return parser.parse_args()