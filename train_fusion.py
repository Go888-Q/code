import os
import argparse
import datetime
import warnings

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from prompt_dataset import PromptDataSet
from model import TextFuse as create_model
from utils import train_one_epoch, evaluate, create_lr_scheduler, load_text_encoder

warnings.filterwarnings("ignore", category=UserWarning)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_dist_initialized()) or dist.get_rank() == 0


def setup_distributed(args):
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    args.use_ddp = args.use_ddp or world_size_env > 1

    if not args.use_ddp:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return device, 0, 1

    if world_size_env <= 1 and "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "DDP was requested, but this script was not launched with torchrun. "
            "Use: torchrun --nproc_per_node=8 train_fusion.py"
        )

    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    device = torch.device("cuda", local_rank)
    return device, rank, world_size


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def create_experiment_dirs():
    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TextIF_train_{}".format(file_name)
    file_img_path = os.path.join(filefold_path, "img")
    file_weights_path = os.path.join(filefold_path, "weights")
    file_log_path = os.path.join(filefold_path, "log")
    file_train_weights_path = os.path.join(filefold_path, "train_weights")

    os.makedirs(filefold_path)
    os.makedirs(file_img_path)
    os.makedirs(file_weights_path)
    os.makedirs(file_log_path)
    os.makedirs(file_train_weights_path)

    return filefold_path, file_img_path, file_weights_path, file_log_path, file_train_weights_path


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def main(args):
    device, rank, world_size = setup_distributed(args)

    tb_writer = None
    if is_main_process():
        filefold_path, file_img_path, file_weights_path, file_log_path, file_train_weights_path = create_experiment_dirs()
        tb_writer = SummaryWriter(log_dir=file_log_path)
    else:
        filefold_path = file_img_path = file_weights_path = file_log_path = file_train_weights_path = None

    if is_dist_initialized():
        shared_paths = [filefold_path, file_img_path, file_weights_path, file_log_path, file_train_weights_path]
        dist.broadcast_object_list(shared_paths, src=0)
        filefold_path, file_img_path, file_weights_path, file_log_path, file_train_weights_path = shared_paths

    best_val_loss = 1e5
    start_epoch = 0

    train_dataset = PromptDataSet("train")
    val_dataset = PromptDataSet("eval")

    batch_size = args.batch_size
    nw = min([os.cpu_count() or 0, batch_size if batch_size > 1 else 0, 8])

    if is_main_process():
        print("Using {} dataloader workers every process".format(nw))
        print("Using distributed training: {} (world size = {})".format(args.use_ddp, world_size))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if args.use_ddp else None

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=nw,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        drop_last=True,
    )

    tokenizer, model_clip, text_model_source = load_text_encoder(
        args.text_model_name,
        device,
        explicit_path=args.text_model_path,
        load_on_cpu=args.text_encoder_on_cpu,
    )
    if is_main_process():
        print("Using text encoder source: {}".format(text_model_source))

    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        if is_main_process():
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            model.load_state_dict(weights_dict, strict=False)

    if args.use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if is_main_process():
            print("Resumed from checkpoint: {}".format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_ssim_loss, train_ssim_loss_mask, train_consist_loss, \
            train_consist_loss_mask, train_text_loss, train_text_loss_mask, lr = train_one_epoch(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                data_loader=train_loader,
                lr_scheduler=lr_scheduler,
                device=device,
                epoch=epoch,
                is_main_process=is_main_process(),
                world_size=world_size,
                scaler=scaler,
                use_amp=use_amp,
            )

        if is_main_process():
            tb_writer.add_scalar("train_total_loss", train_loss, epoch)
            tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
            tb_writer.add_scalar("train_ssim_loss_mask", train_ssim_loss_mask, epoch)
            tb_writer.add_scalar("train_consist_loss", train_consist_loss, epoch)
            tb_writer.add_scalar("train_consist_loss_mask", train_consist_loss_mask, epoch)
            tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)
            tb_writer.add_scalar("train_text_loss_mask", train_text_loss_mask, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0 and is_main_process():
            val_loss, val_ssim_loss, val_ssim_loss_mask, val_consist_loss, val_consist_loss_mask, val_text_loss, val_text_loss_mask, lr = evaluate(
                model=model,
                tokenizer=tokenizer,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                lr=lr,
                filefold_path=file_img_path,
                is_main_process=True,
            )

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss_mask", val_ssim_loss_mask, epoch)
            tb_writer.add_scalar("val_consist_loss", val_consist_loss, epoch)
            tb_writer.add_scalar("val_consist_loss_mask", val_consist_loss_mask, epoch)
            tb_writer.add_scalar("val_text_loss", val_text_loss, epoch)
            tb_writer.add_scalar("val_text_loss_mask", val_text_loss_mask, epoch)

            base_model = unwrap_model(model)
            save_file = {
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(save_file, file_train_weights_path + "/" + str(epoch) + "checkpoint.pth")

            if val_loss < best_val_loss:
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_val_loss = val_loss

            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")

        if is_dist_initialized():
            dist.barrier()

    if tb_writer is not None:
        tb_writer.close()

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed by torchrun')
    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0, help='local rank passed by torchrun')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--val_every_epcho', type=int, default=5, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', action='store_true', help='legacy arg, ignored when using ddp')
    parser.add_argument('--use_ddp', action='store_true', help='use DDP with torchrun')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0,1,2,3,4,5,6,7', help='visible device ids for DDP')
    parser.add_argument('--dist-backend', default='nccl', help='distributed backend')
    parser.add_argument('--text-model-name', default='bert-base-uncased', help='huggingface text encoder name')
    parser.add_argument('--text-model-path', default='./bert-base-uncased', help='local path to a downloaded Hugging Face text encoder')
    parser.add_argument('--text-encoder-on-cpu', action='store_true', default=True, help='keep the frozen text encoder on CPU to save GPU memory')
    opt = parser.parse_args()

    main(opt)
