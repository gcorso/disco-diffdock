import copy
import math
import os
from functools import partial

import wandb
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.autograd.set_detect_anomaly(True)
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from datasets_utils.pdbbind import construct_loader
from utils.parsing import parse_train_args
from utils.training import train_epoch, test_epoch, loss_function, inference_epoch
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, ExponentialMovingAverage
from utils.model_utils import get_model


def train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_earlystop_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    loss_fn = partial(loss_function, tr_weight=args.tr_weight, rot_weight=args.rot_weight,
                      tor_weight=args.tor_weight, no_torsion=args.no_torsion)
    score_model = model.module.score_model if device.type == 'cuda' else model.score_model

    print("Starting training...")
    for epoch in range(args.n_epochs):
        if epoch % 5 == 0: print("Run name: ", args.run_name)
        logs = {}
        train_losses = train_epoch(model, train_loader, optimizer, device, t_to_sigma, loss_fn, ema_weights,
                                   use_latent=args.latent_dim>0, training_latent_temperature=args.training_latent_temperature)
        print("Epoch {}: Training loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
              .format(epoch, train_losses['loss'], train_losses['tr_loss'], train_losses['rot_loss'],
                      train_losses['tor_loss']))

        ema_weights.store(score_model.parameters())
        if args.use_ema: ema_weights.copy_to(score_model.parameters()) # load ema parameters into model for running validation and inference
        val_losses = test_epoch(model, val_loader, device, t_to_sigma, loss_fn, args.test_sigma_intervals, use_latent=args.latent_dim>0,
                                training_latent_temperature=args.training_latent_temperature)
        print("Epoch {}: Validation loss {:.4f}  tr {:.4f}   rot {:.4f}   tor {:.4f}"
              .format(epoch, val_losses['loss'], val_losses['tr_loss'], val_losses['rot_loss'], val_losses['tor_loss']))

        if args.val_inference_freq != None and (epoch + 1) % args.val_inference_freq == 0:
            dataset = [val_loader.dataset.get(i) for i in range(min(args.num_inference_complexes, len(val_loader.dataset)))]
            inf_metrics = inference_epoch(model, dataset, device, t_to_sigma, args, use_latent=args.latent_dim>0)
            print("Epoch {}: Val inference oracle rmsds_lt2 {:.3f} rmsds_lt5 {:.3f}"
                  .format(epoch, inf_metrics['rmsds_lt2'], inf_metrics['rmsds_lt5']))
            logs.update({'valinf_' + k: v for k, v in inf_metrics.items()}, step=epoch + 1)

        if not args.use_ema: ema_weights.copy_to(score_model.parameters())
        ema_state_dict = copy.deepcopy(score_model.state_dict())
        ema_weights.restore(score_model.parameters())

        if args.wandb:
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs['current_lr'] = optimizer.param_groups[0]['lr']
            wandb.log(logs, step=epoch + 1)

        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()
        if args.inference_earlystop_metric in logs.keys() and \
                (args.inference_earlystop_goal == 'min' and logs[args.inference_earlystop_metric] <= best_val_inference_value or
                 args.inference_earlystop_goal == 'max' and logs[args.inference_earlystop_metric] >= best_val_inference_value):
            best_val_inference_value = logs[args.inference_earlystop_metric]
            best_val_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_epoch_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_inference_epoch_model.pt'))
        if val_losses['loss'] <= best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

        if scheduler:
            if args.val_inference_freq is not None:
                scheduler.step(best_val_inference_value)
            else:
                scheduler.step(val_losses['loss'])

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation Loss {} on Epoch {}".format(best_val_loss, best_epoch))
    print("Best inference metric {} on Epoch {}".format(best_val_inference_value, best_val_inference_epoch))


def main_function():
    args = parse_train_args()
    assert (args.inference_earlystop_goal == 'max' or args.inference_earlystop_goal == 'min')
    if args.val_inference_freq is not None and args.scheduler is not None:
        assert (args.scheduler_patience > args.val_inference_freq) # otherwise we will just stop training after args.scheduler_patience epochs
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # construct loader
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.inference_earlystop_goal if args.val_inference_freq is not None else 'min')
    ema_weights = ExponentialMovingAverage(model.module.score_model.parameters() if device.type == 'cuda' else model.score_model.parameters(), decay=args.ema_rate)

    if args.restart_dir:
        try:
            dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
            
            dict['model'] = {k.replace('module.', ''):v for k,v in dict['model'].items()}

            if device.type == 'cuda':
                load_return = model.module.load_state_dict(dict['model'], strict=False)
            else:
                load_return = model.load_state_dict(dict['model'], strict=False)
            print("Loaded model with", load_return)
            
            if len(load_return.missing_keys) == 0 and len(load_return.unexpected_keys) == 0:
                if args.restart_lr is not None: dict['optimizer']['param_groups'][0]['lr'] = args.restart_lr
                optimizer.load_state_dict(dict['optimizer'])
                if hasattr(args, 'ema_rate'):
                    ema_weights.load_state_dict(dict['ema_weights'], device=device)
                print("Restarting from epoch", dict['epoch'])
            else:
                assert args.non_strict_loading, f"Issue with loading of the model: missing keys {load_return.missing_keys} unexpected_keys {load_return.unexpected_keys}"
                    
        except Exception as e:
            raise e 
            print("Exception", e)
            dict = torch.load(f'{args.restart_dir}/best_model.pt', map_location=torch.device('cpu'))
            model.load_state_dict(dict, strict=True)
            print("Due to exception had to take the best epoch and no optimiser")

    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters in total')

    if args.wandb:
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main_function()