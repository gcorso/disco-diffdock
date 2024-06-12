from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch_geometric.nn.data_parallel import DataParallel

from models.all_atom_score_model import TensorProductScoreModel as AAScoreModel
from models.latent_encoder import TPEncoder
from models.model_classes import ModelWrapper
from models.pretrained_score_encoder import PretrainedScoreEncoder
from models.score_model import TensorProductScoreModel as CGScoreModel
from utils.diffusion_utils import get_timestep_embedding
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


def remove_data_parallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k2 = k.replace("module.", "")   # remove `module.`
        new_state_dict[k2] = v
    return new_state_dict


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False):
    if 'all_atoms' in args and args.all_atoms:
        model_class = AAScoreModel
    else:
        model_class = CGScoreModel

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    score_model = model_class(t_to_sigma=t_to_sigma,
                              device=device,
                              no_torsion=args.no_torsion,
                              timestep_emb_func=timestep_emb_func,
                              num_conv_layers=args.num_conv_layers,
                              lig_max_radius=args.max_radius,
                              scale_by_sigma=args.scale_by_sigma,
                              sigma_embed_dim=args.sigma_embed_dim,
                              ns=args.ns, nv=args.nv,
                              distance_embed_dim=args.distance_embed_dim,
                              cross_distance_embed_dim=args.cross_distance_embed_dim,
                              batch_norm=not args.no_batch_norm,
                              dropout=args.dropout,
                              sh_lmax=args.sh_lmax if hasattr(args, 'sh_lmax') else 2,
                              use_second_order_repr=args.use_second_order_repr,
                              cross_max_distance=args.cross_max_distance,
                              dynamic_max_cross=args.dynamic_max_cross,
                              lm_embedding_type=lm_embedding_type,
                              confidence_mode=confidence_mode,
                              num_confidence_outputs=len(
                                  args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                  args.rmsd_classification_cutoff, list) else 1,
                              use_old_atom_encoder=args.use_old_atom_encoder if hasattr(args, 'use_old_atom_encoder') else True,
                              latent_dim=args.latent_dim if hasattr(args, 'latent_dim') else 0,
                              latent_vocab=args.latent_vocab if hasattr(args, 'latent_vocab') else 0,
                              latent_cross_attention=args.latent_cross_attention if hasattr(args, 'latent_cross_attention') else False,
                              new_cross_attention=args.new_cross_attention if hasattr(args, 'new_cross_attention') else False,
                              cross_attention_heads=args.cross_attention_heads if hasattr(args, 'cross_attention_heads') else 1,
                              cross_attention_dim=args.cross_attention_dim if hasattr(args, 'cross_attention_dim') else 64,
                              latent_droprate=args.latent_droprate if hasattr(args, 'latent_droprate') else 0)

    encoder = None
    if hasattr(args, 'latent_vocab') and args.latent_dim > 0:
        encoder = TPEncoder(device=device,
                            no_torsion=args.no_torsion,
                            num_conv_layers=args.encoder_num_conv_layers,
                            lig_max_radius=args.max_radius,
                            ns=args.encoder_ns, nv=args.encoder_nv,
                            distance_embed_dim=args.distance_embed_dim,
                            cross_distance_embed_dim=args.cross_distance_embed_dim,
                            batch_norm=not args.no_batch_norm,
                            dropout=args.dropout,
                            sh_lmax=args.sh_lmax if hasattr(args, 'sh_lmax') else 2,
                            use_second_order_repr=args.use_second_order_repr,
                            cross_max_distance=args.encoder_cross_max_distance,
                            lm_embedding_type=lm_embedding_type if not args.encoder_no_esm else None,
                            latent_dim=args.latent_dim,
                            latent_vocab=args.latent_vocab,
                            latent_no_batchnorm=args.latent_no_batchnorm,
                            latent_dropout=args.latent_dropout,
                            latent_hidden_dim=args.latent_hidden_dim,
                            latent_virtual_nodes=args.latent_virtual_nodes,
                            latent_nodes_residual=args.latent_nodes_residual)

    if hasattr(args, 'latent_vocab'):
        model = ModelWrapper(encoder=encoder, score_model=score_model, training_latent_temperature=args.training_latent_temperature, device=device, latent_droprate=args.latent_droprate if hasattr(args, 'latent_droprate') else 0)
    else:
        model = score_model

    if device.type == 'cuda' and not no_parallel:
        model = DataParallel(model)
    model.to(device)
    return model


def get_ar_model(args, score_model_args, device, training=True):
    lm_embedding_type = None
    if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    if 'use_pretrained_score' in args and args.use_pretrained_score:
        t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
        model = get_model(score_model_args, device, t_to_sigma)
        model = model.module if isinstance(model, nn.DataParallel) else model
        state_dict = torch.load(f'{args.original_model_dir}/{args.ckpt}', map_location=device)
        model.load_state_dict(state_dict, strict=True)
        score_model = model.score_model

        model = PretrainedScoreEncoder(pretrained_score_model=score_model,
                                       ns=args.ns,
                                       latent_dim=1,
                                       latent_vocab=score_model_args.latent_vocab,
                                       latent_no_batchnorm=args.latent_no_batchnorm,
                                       latent_dropout=args.latent_dropout,
                                       latent_hidden_dim=args.latent_hidden_dim,
                                       input_latent_dim=score_model_args.latent_dim,
                                       apply_gumbel_softmax=False if training else True)

    else:
        model = TPEncoder(device=device,
                          no_torsion=args.no_torsion,
                          num_conv_layers=args.num_conv_layers,
                          lig_max_radius=args.max_radius,
                          ns=args.ns, nv=args.nv,
                          distance_embed_dim=args.distance_embed_dim,
                          cross_distance_embed_dim=args.cross_distance_embed_dim,
                          batch_norm=not args.no_batch_norm,
                          dropout=args.dropout,
                          sh_lmax=args.sh_lmax if hasattr(args, 'sh_lmax') else 2,
                          use_second_order_repr=args.use_second_order_repr,
                          cross_max_distance=args.cross_max_distance,
                          lm_embedding_type=lm_embedding_type,
                          latent_dim=1,
                          latent_vocab=score_model_args.latent_vocab,
                          latent_no_batchnorm=args.latent_no_batchnorm,
                          latent_dropout=args.latent_dropout,
                          latent_hidden_dim=args.latent_hidden_dim,
                          use_oracle=False,
                          input_latent_dim=score_model_args.latent_dim,
                          apply_gumbel_softmax=False if training else True,
                          latent_virtual_nodes=args.latent_virtual_nodes,
                          latent_nodes_residual=args.latent_nodes_residual)

    model = model.to(device)
    return model
