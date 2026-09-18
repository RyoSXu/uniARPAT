import os
import time
import argparse
import json
import random
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from utils.builder import ConfigBuilder
from model.model import basemodel
from utils.experiment_config import ExperimentConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ablation')


def setup_ablation_seed(seed: int):
    """H1 hygiene: deterministic seeding for perfect对照 (init + shuffle + cudnn).

    NOTE: torch.backends.cudnn.deterministic=True costs speed; enabled here
    because ablation comparability outranks throughput (V100 has headroom).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL_CONFIGS = {
    'M1': {
        'desc': 'Baseline: Shared Decoder, Asymmetric Heads (1-layer eDOS / 6-layer phDOS), Oracle Scale (77.63M params)',
        'transformer_params': {
            'decoupled_decoder': False,
            'use_gated_cross_attn': False,
            'head_type': 'legacy',
            'predict_scale': False
        }
    },
    'M2': {
        'desc': 'Decoupled Decoder, Trimmed phDOS Head (3-layer 0.788M), Oracle Scale (72.96M params, -4.67M vs M1)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'ph_trimmed',
            'predict_scale': False
        }
    },
    'M3': {
        'desc': 'Decoupled Decoder, Capacity Symmetric Heads (0.788M each), Oracle Scale (73.75M params, -3.88M vs M1)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M4': {
        'desc': 'Decoupled Decoder, Symmetric Heads, Post-Decoder Zero-Init Gated Cross Attention (75.85M params, +2.10M vs M3)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M5': {
        'desc': 'Full uniARPAT: Decoupled, Symmetric, Gated Attention, Shape-Scale Decoupling & Physical Loss (75.93M params)',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': True
        }
    }
}

def _resolve_edos_grid(spec: str, edos_num: int):
    """C1.2: grid key (E0/E2..) in grids.json or path to centers npy -> centers list."""
    import os as _os
    if _os.path.exists(spec):
        return np.load(spec).tolist()
    with open('./data/grids_c2b/grids.json') as f:
        grids = json.load(f)
    if spec in grids:
        e = np.asarray(grids[spec], dtype=float)
        assert len(e) - 1 == edos_num, f"{spec} bins {len(e)-1} != {edos_num}"
        return ((e[:-1] + e[1:]) / 2).tolist()
    raise ValueError(f"unknown edos_grid spec: {spec}")


def _ph_grid_centers(phdos_num: int):
    """C2b: bin centers for the P-arm matching phdos_num (P0/P1/P2 in grids.json)."""
    try:
        with open('./data/grids_c2b/grids.json') as f:
            grids = json.load(f)
    except Exception:
        return None
    for arm in ("P0", "P1", "P2"):
        e = np.asarray(grids.get(arm, []), dtype=float)
        if len(e) - 1 == phdos_num:
            return ((e[:-1] + e[1:]) / 2).tolist()
    return None


def train_and_eval(cfg: ExperimentConfig):
    if cfg.model_name not in MODEL_CONFIGS :
        raise ValueError (f"Unknown model name: {cfg.model_name }. Available: {list (MODEL_CONFIGS .keys ())}")

    setup_ablation_seed (cfg.seed )
    # B3: tag isolates variant runs (e.g. batch-size bridge) from h1 outputs.
    suffix =cfg.model_name .lower ()+cfg.tag 
    summary_file =f"./results/test_{suffix }_summary.csv"
    if cfg.skip_existing and os .path .exists (summary_file ):
        logger .info (f"[{cfg.model_name }] Already completed ({summary_file } exists). Skipping.")
        return pd .read_csv (summary_file ).to_dict (orient ='records')[0 ]

    config_info =MODEL_CONFIGS [cfg.model_name ]
    logger .info ("="*70 )
    logger .info (f"   STARTING ABLATION EXPERIMENT: {cfg.model_name }")
    logger .info (f"   Description: {config_info ['desc']}")
    logger .info (f"   Epochs: {cfg.epochs } | Batch Size: {cfg.batch_size } | LR: {cfg.lr } | Seed: {cfg.seed }")
    logger .info ("="*70 )

    save_dir =f"./output/ablation_{suffix }"
    os .makedirs (save_dir ,exist_ok =True )
    os .makedirs ('./results',exist_ok =True )

    with open ('configs/default.yaml')as f :
        yaml_cfg =yaml .load (f ,Loader =yaml .FullLoader )

    yaml_cfg ['model']['params']['sub_model']['transformer'].update (config_info ['transformer_params'])
    # C2b: grid arms override output dims + data source (defaults = v1/h1 behavior).
    yaml_cfg ['model']['params']['sub_model']['transformer']['edos_num']=cfg.edos_num 
    yaml_cfg ['model']['params']['sub_model']['transformer']['phdos_num']=cfg.phdos_num 
    yaml_cfg ['model']['params']['sub_model']['transformer']['atom_feat_mode']=cfg.atom_feat 
    yaml_cfg ['model']['params']['sub_model']['transformer']['energy_code']=cfg.energy_code 
    for _k ,_v in (("tv_w",cfg.tv_w ),("grad_w",cfg.grad_w ),("peak_w",cfg.peak_w ),
    ("tail_w",cfg.tail_w ),("tail_start",cfg.tail_start )):
        yaml_cfg ['model']['params'][_k ]=_v 
        # C2.1: sumnorm norm => KL/W+Huber loss form (dataset flag mirrored here).
    yaml_cfg ['model']['params']['use_mask']=bool (cfg.use_mask )
    # C2.4 decoupled scale head (default off).
    yaml_cfg ['model']['params']['sub_model']['transformer']['scale_mode']=cfg.scale_mode 
    yaml_cfg ['model']['params']['scale_sup_w']=float (cfg.scale_sup_w )
    yaml_cfg ['model']['params']['eta_sup_w']=float (cfg.eta_sup_w )
    yaml_cfg ['model']['params']['delta_edos']=float (cfg.delta_edos )
    yaml_cfg ['model']['params']['delta_phdos']=float (cfg.delta_phdos )
    # S1 boundary scalars (default off).
    yaml_cfg ['model']['params']['sub_model']['transformer']['scalar_mode']=cfg.scalar_mode 
    yaml_cfg ['model']['params']['scalar_sup_w']=float (cfg.scalar_sup_w )
    # E9-P0 G1 exact sparse graph (default off: off-path bit-identical).
    yaml_cfg ['model']['params']['sub_model']['transformer']['use_g1']=bool (cfg.use_g1 )
    yaml_cfg ['model']['params']['sub_model']['transformer']['g1_r_cut']=float (cfg.g1_r_cut )
    yaml_cfg ['model']['params']['sub_model']['transformer']['g1_max_neighbors']=int (cfg.g1_max_neighbors )
    # E9-P0 Q1 coordinate trunks (default off).
    yaml_cfg ['model']['params']['sub_model']['transformer']['q1_coord']=bool (cfg.q1_coord )
    yaml_cfg ['model']['params']['sub_model']['transformer']['q1_hidden']=int (cfg.q1_hidden )
    # E9-P0 Q2 Fourier trunk (default off; implies the trunk pathway).
    yaml_cfg ['model']['params']['sub_model']['transformer']['q2_fourier']=bool (cfg.q2_fourier )
    # B4 hyperparams (None = config default, preserves legacy behavior).
    if cfg.dropout is not None :
        yaml_cfg ['model']['params']['sub_model']['transformer']['dropout']=float (cfg.dropout )
    if cfg.weight_decay is not None :
        yaml_cfg ['model']['params']['optimizer']['transformer']['params']['weight_decay']=float (cfg.weight_decay )
    if cfg.lambda_ph is not None :
        yaml_cfg ['model']['params']['lambda_ph']=float (cfg.lambda_ph )
    if cfg.grad_clip is not None :
        yaml_cfg ['model']['params']['grad_clip']=float (cfg.grad_clip )
        # L3: KL/W1/Huber term ablation (None = config default 1.0/1.0, B4-style).
    if cfg.w_w1 is not None :
        yaml_cfg ['model']['params']['w_w1']=float (cfg.w_w1 )
    if cfg.w_huber is not None :
        yaml_cfg ['model']['params']['w_huber']=float (cfg.w_huber )
    _wu =cfg.warmup_epochs 
    yaml_cfg ['model']['params']['loss_form']="sumnorm_klw"if cfg.norm =="sumnorm"else "smoothl1"
    _sn =(cfg.norm =="sumnorm")
    yaml_cfg ['model']['params']['sub_model']['transformer']['energy_code']=cfg.energy_code 
    if cfg.energy_code =="edos":
        yaml_cfg ['model']['params']['sub_model']['transformer']['edos_grid']=_resolve_edos_grid (cfg.edos_grid ,cfg.edos_num )
    yaml_cfg ['model']['params']['dos_minmax']=True 
    yaml_cfg ['model']['params']['save_best']='balanced_score'
    yaml_cfg ['dataset']['train']['data_dir']=cfg.data_dir 
    yaml_cfg ['dataset']['valid']['data_dir']=cfg.data_dir 
    yaml_cfg ['dataset']['test']['data_dir']=cfg.data_dir 
    # C1.4: aug flags ride the train dict into Dos_Dataset (valid/test clean).
    yaml_cfg ['dataset']['train']['augment']=bool (cfg.augment )
    yaml_cfg ['dataset']['train']['disp_sigma']=float (cfg.disp_sigma )

    # E1 hygiene: dump effective config (reproducibility; train.py already does this).
    with open (os .path .join (save_dir ,'config_used.yaml'),'w')as f :
        yaml .dump ({'cli':{'model':cfg.model_name ,'epochs':cfg.epochs ,
        'batch_size':cfg.batch_size ,'lr':cfg.lr ,'seed':cfg.seed ,
        'data_dir':cfg.data_dir ,'edos_num':cfg.edos_num ,
        'phdos_num':cfg.phdos_num ,'atom_feat':cfg.atom_feat ,
        'energy_code':cfg.energy_code ,'edos_grid':cfg.edos_grid ,
        'tv_w':cfg.tv_w ,'grad_w':cfg.grad_w ,'peak_w':cfg.peak_w ,
        'tail_w':cfg.tail_w ,'tail_start':cfg.tail_start ,
        'augment':cfg.augment ,'disp_sigma':cfg.disp_sigma ,
        'norm':cfg.norm ,'use_mask':cfg.use_mask ,'dropout':cfg.dropout ,
        'weight_decay':cfg.weight_decay ,'warmup_epochs':_wu ,
        'lambda_ph':cfg.lambda_ph ,'grad_clip':cfg.grad_clip ,
        'w_w1':cfg.w_w1 ,'w_huber':cfg.w_huber ,
        'scale_mode':cfg.scale_mode ,'freeze_backbone':cfg.freeze_backbone ,
        'eta_sup_w':cfg.eta_sup_w ,
        'delta_edos':cfg.delta_edos ,'delta_phdos':cfg.delta_phdos ,
        'scalar_mode':cfg.scalar_mode ,'scalar_sup_w':cfg.scalar_sup_w ,
        'use_g1':cfg.use_g1 ,'g1_r_cut':cfg.g1_r_cut ,
        'g1_max_neighbors':cfg.g1_max_neighbors ,
        'q1_coord':cfg.q1_coord ,'q1_hidden':cfg.q1_hidden ,
        'q2_fourier':cfg.q2_fourier ,
        'init_ckpt':cfg.init_ckpt ,'scale_sup_w':cfg.scale_sup_w },
        'config':yaml_cfg },f ,indent =2 ,sort_keys =False ,
        default_flow_style =False )

    builder =ConfigBuilder (**yaml_cfg )

    train_loader =builder .get_dataloader (split ='train',dos_minmax =True ,batch_size =cfg.batch_size ,dos_sumnorm =_sn )
    val_loader =builder .get_dataloader (split ='valid',dos_minmax =True ,batch_size =cfg.batch_size ,dos_sumnorm =_sn )
    test_loader =builder .get_dataloader (split ='test',dos_minmax =True ,batch_size =cfg.batch_size ,dos_sumnorm =_sn )

    model =builder .get_model ()
    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    model .to (device )

    total_params =sum (p .numel ()for p in model .model ['transformer'].parameters ()if p .requires_grad )
    logger .info (f"[{cfg.model_name }] Verified Trainable Parameters: {total_params :,} ({total_params /1e6 :.3f}M)")

    # C2.4 Phase A: init from trained backbone (strict=False tolerates new head),
    # optionally freeze everything except the scale head.
    if cfg.init_ckpt :
        _ck =torch .load (cfg.init_ckpt ,map_location ='cpu')
        _st =_ck ['model']if isinstance (_ck ,dict )and 'model'in _ck else _ck 
        _miss ,_unexp =model .model ['transformer'].load_state_dict (_st ,strict =False )
        logger .info (f"[{cfg.model_name }] init_ckpt loaded: missing={list (_miss )[:5 ]} unexpected={list (_unexp )[:5 ]}")
        model .to (device )
    if cfg.freeze_backbone :
        model .model ['transformer'].requires_grad_ (False )
        ntr =0 
        for n ,p in model .model ['transformer'].named_parameters ():
            if 'scale_head_c24'in n or 'eta_head'in n or 'scalar_head'in n :
                p .requires_grad_ (True )
                ntr +=p .numel ()
        logger .info (f"[{cfg.model_name }] backbone frozen, trainable scale params: {ntr :,}")

    optimizer =model .optimizer ['transformer']
    # B3 fix (latent bug): --lr was accepted but never applied (optimizer kept
    # config lr). Apply CLI lr BEFORE scheduler construction.
    for pg in optimizer .param_groups :
        pg ['lr']=cfg.lr 
        pg ['initial_lr']=cfg.lr 
    logger .info (f"[{cfg.model_name }] Effective optimizer LR set to {cfg.lr :.2e}")
    # H4 hygiene: warmup+cosine shared with train.py semantics (was bare cosine).
    from utils .builder import build_warmup_cosine_scheduler 
    scheduler =build_warmup_cosine_scheduler (optimizer ,cfg.epochs )if _wu is None else build_warmup_cosine_scheduler (optimizer ,cfg.epochs ,warmup_epochs =int (_wu ))

    best_val_score =float ('inf')
    best_epoch =0 
    history =[]
    start_epoch =0 

    # Resume-from-latest: long runs may be killed by infra; resume losslessly.
    # checkpoint_latest.pth carries {epoch, model, optimizer, best_val_score}.
    latest_p =os .path .join (save_dir ,'checkpoint_latest.pth')
    hist_p =f"./results/history_{suffix }.csv"
    if os .path .exists (latest_p ):
        try :
            ck =torch .load (latest_p ,map_location ='cpu')
            state =ck ['model']if isinstance (ck ,dict )and 'model'in ck else ck 
            model .model ['transformer'].load_state_dict (state )
            if isinstance (ck ,dict )and 'optimizer'in ck :
                try :
                    optimizer .load_state_dict (ck ['optimizer'])
                except Exception :
                    pass 
            start_epoch =int (ck .get ('epoch',0 ))if isinstance (ck ,dict )else 0 
            best_val_score =float (ck .get ('best_val_score',float ('inf')))if isinstance (ck ,dict )else float ('inf')
            # restore best_epoch + history (history file optional: killed runs
            # only have checkpoints; it is rewritten incrementally below).
            if os .path .exists (hist_p ):
                dfh =pd .read_csv (hist_p )
                history =dfh .to_dict (orient ='records')
                if start_epoch <=0 and len (dfh ):
                    start_epoch =int (dfh ['epoch'].max ())
                if history :
                    best_epoch =int (dfh .loc [dfh ['balanced_score'].idxmin (),'epoch'])
                    # fast-forward cosine part of scheduler to start_epoch
            for _ in range (start_epoch ):
                scheduler .step ()
            model .to (device )
            logger .info (f"[{cfg.model_name }] Resumed from epoch {start_epoch } "
            f"(best ep {best_epoch }, score {best_val_score :.4f})")
        except Exception as e :
            logger .info (f"[{cfg.model_name }] Resume failed ({e }); starting fresh.")
            start_epoch ,history =0 ,[]
            best_val_score ,best_epoch =float ('inf'),0 

    for epoch in range (start_epoch ,cfg.epochs ):
    # H1 hygiene: reshuffle each epoch (DistributedSampler defaults to epoch=0
    # forever when set_epoch is never called -> identical batch order every epoch).
        sampler =getattr (train_loader ,"sampler",None )
        if sampler is not None and hasattr (sampler ,"set_epoch"):
            sampler .set_epoch (epoch )
        model .model ['transformer'].train ()
        train_loss =0.0 
        sub_loss_accum ={}
        n_batches =len (train_loader )
        t_start =time .time ()

        for step ,batch in enumerate (train_loader ):
            loss_dict =model .train_one_step (batch ,step =step )
            train_loss +=loss_dict ['loss']
            for k ,v in loss_dict .items ():
                sub_loss_accum [k ]=sub_loss_accum .get (k ,0.0 )+v 
            if (step +1 )%100 ==0 or (step +1 )==n_batches :
                logger .info (
                f"[{cfg.model_name }] Epoch [{epoch +1 :03d}/{cfg.epochs :03d}] Step [{step +1 :03d}/{n_batches :03d}] | "
                f"Loss: {loss_dict ['loss']:.4f} (eDOS: {loss_dict ['loss_edos']:.4f}, phDOS: {loss_dict ['loss_phdos']:.4f})"
                )

        scheduler .step ()
        torch .cuda .synchronize ()if torch .cuda .is_available ()else None 
        ep_time =time .time ()-t_start 
        train_loss /=n_batches 
        avg_sub_losses ={f"train_{k }":v /n_batches for k ,v in sub_loss_accum .items ()}

        # 显存实测打点 (§6 补正 1):每轮记录峰值显存,写入 history_*.csv 的 peak_vram_mb 列
        peak_vram_mb =torch .cuda .max_memory_allocated ()/(1024 **2 )if torch .cuda .is_available ()else 0.0 

        # Extract gate scalars if gated cross-attention is present
        alpha_e ,alpha_p =0.0 ,0.0 
        if model .model ['transformer'].use_gated_cross_attn :
            alpha_e =model .model ['transformer'].gated_cross_attn .alpha_e .item ()
            alpha_p =model .model ['transformer'].gated_cross_attn .alpha_p .item ()

            # Validation (Dual-track Blind & Oracle for M5)
        val_metrics =evaluate_split (model ,val_loader ,is_m5 =(cfg.model_name =='M5'),ph_grid =_ph_grid_centers (cfg.phdos_num ),sumnorm =(cfg.norm =='sumnorm'))
        balanced =0.5 *val_metrics ['mae_edos_median']+0.5 *val_metrics ['mae_phdos_median']

        log_str =(
        f"[{cfg.model_name }] Epoch [{epoch +1 :03d}/{cfg.epochs :03d}] ({ep_time :.1f}s) | "
        f"Train Loss: {train_loss :.4f} | "
        f"Val eDOS R2(med): {val_metrics ['r2_edos_median']:.3f} (MAE: {val_metrics ['mae_edos_median']:.3f}) | "
        f"Val phDOS R2(med): {val_metrics ['r2_phdos_median']:.3f} (MAE: {val_metrics ['mae_phdos_median']:.4f}) | "
        f"Balanced Score: {balanced :.4f}"
        )
        if cfg.model_name =='M5':
            log_str +=f" | Oracle eDOS R2: {val_metrics ['oracle_r2_edos_median']:.3f}, phDOS R2: {val_metrics ['oracle_r2_phdos_median']:.3f}"
        if model .model ['transformer'].use_gated_cross_attn :
            log_str +=f" | Gate (a_e={alpha_e :.4f}, a_p={alpha_p :.4f})"
        logger .info (log_str )

        history .append ({
        'epoch':epoch +1 ,
        'seed':cfg.seed ,
        'epoch_time_s':ep_time ,
        'peak_vram_mb':peak_vram_mb ,
        'alpha_e':alpha_e ,
        'alpha_p':alpha_p ,
        **avg_sub_losses ,
        **val_metrics ,
        'balanced_score':balanced 
        })

        # 如需逐轮独立峰值则重置计数器,下一轮重新统计
        if torch .cuda .is_available ():
            torch .cuda .reset_peak_memory_stats ()

            # E2 hygiene: full checkpoint dict (was bare state_dict), unified with
            # train.py format. Loader below + cif2dos both accept this format.
            # Atomic write (tmp + rename): concurrent/duplicate runners or SIGKILL
            # mid-save must never leave a torn checkpoint behind.
        def _ckpt (epoch_ ,best_ ):
            return {'epoch':epoch_ ,
            'model_name':cfg.model_name ,
            'seed':cfg.seed ,
            'model':model .model ['transformer'].state_dict (),
            'optimizer':optimizer .state_dict (),
            'best_val_score':best_ }

        def _atomic_save (obj ,path ):
            tmp =path +'.tmp'
            torch .save (obj ,tmp )
            os .replace (tmp ,path )

        if balanced <best_val_score :
            best_val_score =balanced 
            best_epoch =epoch +1 
            _atomic_save (_ckpt (epoch +1 ,balanced ),os .path .join (save_dir ,'checkpoint_best.pth'))
            logger .info (f"[{cfg.model_name }] New best model saved at Epoch {epoch +1 } (Score: {balanced :.4f})")

        _atomic_save (_ckpt (epoch +1 ,best_val_score ),os .path .join (save_dir ,'checkpoint_latest.pth'))
        # Incremental history (killed runs resume from checkpoint + history).
        pd .DataFrame (history ).to_csv (f"./results/history_{suffix }.csv",index =False )

        # Save training history
    df_history =pd .DataFrame (history )
    df_history .to_csv (f"./results/history_{suffix }.csv",index =False )

    # Load best checkpoint and evaluate on Test set
    logger .info (f"\nEvaluating Best Model ({cfg.model_name }, Epoch {best_epoch }) on Test Set (1371 materials)...")
    best_ckpt =torch .load (os .path .join (save_dir ,'checkpoint_best.pth'))
    best_state =best_ckpt ['model']if isinstance (best_ckpt ,dict )and 'model'in best_ckpt else best_ckpt 
    model .model ['transformer'].load_state_dict (best_state )

    test_metrics =evaluate_split (model ,test_loader ,is_m5 =(cfg.model_name =='M5'),return_sample_level =True ,
    ph_grid =_ph_grid_centers (cfg.phdos_num ),sumnorm =(cfg.norm =='sumnorm'))
    df_samples =test_metrics .pop ('sample_df')
    pred_p =test_metrics .pop ('pred_phdos')
    pred_e =test_metrics .pop ('pred_edos')
    tgt_p =test_metrics .pop ('tgt_phdos')
    tgt_e =test_metrics .pop ('tgt_edos')

    np .save (f"./results/pred_phdos_{suffix }.npy",pred_p )
    np .save (f"./results/pred_edos_{suffix }.npy",pred_e )
    if cfg.model_name =='M5':
        np .save ('./results/pred_phdos.npy',pred_p )
        np .save ('./results/pred_edos.npy',pred_e )
        np .save ('./results/tgt_phdos.npy',tgt_p )
        np .save ('./results/tgt_edos.npy',tgt_e )
        df_samples .to_csv ('./results/test_evaluation_summary.csv',index =False )

    df_samples .to_csv (f"./results/samples_{suffix }_test.csv",index =False )

    df_test_summary =pd .DataFrame ([test_metrics ])
    df_test_summary .to_csv (f"./results/test_{suffix }_summary.csv",index =False )

    logger .info ("="*70 )
    logger .info (f"   TEST EVALUATION SUMMARY ({cfg.model_name })")
    logger .info ("="*70 )
    for k ,v in test_metrics .items ():
        logger .info (f"   {k :28s}: {v :.4f}"if isinstance (v ,float )else f"   {k :28s}: {v }")
    logger .info ("="*70 )
    return test_metrics 

def evaluate_split (model ,dataloader ,is_m5 :bool =False ,return_sample_level :bool =False ,ph_grid =None ,sumnorm :bool =False ):
    model .model ['transformer'].eval ()
    records =[]
    # C2b: thermo needs a uniform freq grid; nonuniform arms (P1) skip thermo
    # (verdict metrics are spectral; thermo tracked where defined).
    skip_thermo ,_calc_override =False ,None 
    if ph_grid is not None :
        import numpy as _np 
        _g =_np .asarray (ph_grid ,dtype =float )
        _d =_np .diff (_g )
        if _np .allclose (_d ,_d [0 ],rtol =1e-3 ):
            from thermo_props import ThermodynamicCalculator as _TC 
            _calc_override =_TC (ph_freq_min =float (_g [0 ]),ph_freq_max =float (_g [-1 ]),
            ph_bins =len (_g ))
        else :
            skip_thermo =True 
    if return_sample_level :
        all_p_p ,all_t_p =[],[]
        all_p_e ,all_t_e =[],[]

    with torch .no_grad ():
        for batch in dataloader :
            inp ,pos ,mask ,edos_tgt ,phdos_tgt ,edos_m ,edos_s ,edos_min ,edos_max ,phdos_m ,phdos_s ,phdos_min ,phdos_max ,edos_cov ,phdos_cov ,_nvalence ,edos_x ,phdos_x =model .data_preprocess (batch )

            outputs =model .model ['transformer'](inp ,mask ,pos ,edos_x ,phdos_x )

            # Targets in true physical space
            t_e =edos_tgt *(edos_max -edos_min )+edos_min 
            t_p =phdos_tgt *(phdos_max -phdos_min )+phdos_min 

            if is_m5 :
            # 1. Blind physical prediction
                p_e =torch .clamp (outputs ['phys_edos'],min =0.0 )
                p_p =torch .clamp (outputs ['phys_phdos'],min =0.0 )
                # 2. Oracle denormalization (with ground-truth min/max)
                p_e_orc =torch .clamp (outputs ['shape_edos']*(edos_max -edos_min )+edos_min ,min =0.0 )
                p_p_orc =torch .clamp (outputs ['shape_phdos']*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
            else :
            # Oracle denormalization for M1-M4. C2.1 sumnorm: head emits raw
            # logits (distribution lives behind softmax in loss); eval must
            # softmax first, then scale by sum slots (min=0 here by design).
                if sumnorm :
                    import torch .nn .functional as _F 
                    p_e =torch .clamp (_F .softmax (outputs ['edos'],dim =-1 )*(edos_max -edos_min )+edos_min ,min =0.0 )
                    p_p =torch .clamp (_F .softmax (outputs ['phdos'],dim =-1 )*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
                else :
                    p_e =torch .clamp (outputs ['edos']*(edos_max -edos_min )+edos_min ,min =0.0 )
                    p_p =torch .clamp (outputs ['phdos']*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
                p_e_orc ,p_p_orc =None ,None 

                # Sample-wise primary metrics (H2 hygiene: shared fn, bit-identical math)
            from utils .metrics import per_sample_spectral_metrics 
            _me =per_sample_spectral_metrics (p_e ,t_e )
            mae_e ,mse_e ,r2_e =_me ['mae'],_me ['mse'],_me ['r2']
            _mp =per_sample_spectral_metrics (p_p ,t_p )
            mae_p ,mse_p ,r2_p =_mp ['mae'],_mp ['mse'],_mp ['r2']

            # Sample-wise Oracle metrics if M5
            if is_m5 :
                _meo =per_sample_spectral_metrics (p_e_orc ,t_e )
                mae_e_orc ,mse_e_orc ,r2_e_orc =_meo ['mae'],_meo ['mse'],_meo ['r2']
                _mpo =per_sample_spectral_metrics (p_p_orc ,t_p )
                mae_p_orc ,mse_p_orc ,r2_p_orc =_mpo ['mae'],_mpo ['mse'],_mpo ['r2']

            if return_sample_level :
                all_p_p .append (p_p .cpu ().numpy ())
                all_t_p .append (t_p .cpu ().numpy ())
                all_p_e .append (p_e .cpu ().numpy ())
                all_t_e .append (t_e .cpu ().numpy ())

            B =inp .shape [0 ]
            for i in range (B ):
                rec ={
                'mae_edos':mae_e [i ].item (),
                'mse_edos':mse_e [i ].item (),
                'r2_edos':r2_e [i ].item (),
                'mae_phdos':mae_p [i ].item (),
                'mse_phdos':mse_p [i ].item (),
                'r2_phdos':r2_p [i ].item (),
                }
                if is_m5 :
                    rec .update ({
                    'oracle_mae_edos':mae_e_orc [i ].item (),
                    'oracle_mse_edos':mse_e_orc [i ].item (),
                    'oracle_r2_edos':r2_e_orc [i ].item (),
                    'oracle_mae_phdos':mae_p_orc [i ].item (),
                    'oracle_mse_phdos':mse_p_orc [i ].item (),
                    'oracle_r2_phdos':r2_p_orc [i ].item (),
                    })
                records .append (rec )

    df =pd .DataFrame (records )
    summary ={
    'mae_edos_mean':float (df ['mae_edos'].mean ()),
    'mae_edos_std':float (df ['mae_edos'].std ()),
    'mae_edos_median':float (df ['mae_edos'].median ()),
    'mse_edos_mean':float (df ['mse_edos'].mean ()),
    'mse_edos_std':float (df ['mse_edos'].std ()),
    'r2_edos_mean':float (df ['r2_edos'].mean ()),
    'r2_edos_std':float (df ['r2_edos'].std ()),
    'r2_edos_median':float (df ['r2_edos'].median ()),
    'fail_rate_edos':float ((df ['r2_edos']<0 ).mean ()*100.0 ),

    'mae_phdos_mean':float (df ['mae_phdos'].mean ()),
    'mae_phdos_std':float (df ['mae_phdos'].std ()),
    'mae_phdos_median':float (df ['mae_phdos'].median ()),
    'mse_phdos_mean':float (df ['mse_phdos'].mean ()),
    'mse_phdos_std':float (df ['mse_phdos'].std ()),
    'r2_phdos_mean':float (df ['r2_phdos'].mean ()),
    'r2_phdos_std':float (df ['r2_phdos'].std ()),
    'r2_phdos_median':float (df ['r2_phdos'].median ()),
    'fail_rate_phdos':float ((df ['r2_phdos']<0 ).mean ()*100.0 ),
    }

    if is_m5 :
        summary .update ({
        'oracle_mae_edos_mean':float (df ['oracle_mae_edos'].mean ()),
        'oracle_mae_edos_median':float (df ['oracle_mae_edos'].median ()),
        'oracle_r2_edos_mean':float (df ['oracle_r2_edos'].mean ()),
        'oracle_r2_edos_median':float (df ['oracle_r2_edos'].median ()),
        'oracle_mae_phdos_mean':float (df ['oracle_mae_phdos'].mean ()),
        'oracle_mae_phdos_median':float (df ['oracle_mae_phdos'].median ()),
        'oracle_r2_phdos_mean':float (df ['oracle_r2_phdos'].mean ()),
        'oracle_r2_phdos_median':float (df ['oracle_r2_phdos'].median ()),
        })

    if return_sample_level :
        from thermo_props import ThermodynamicCalculator 
        calc =_calc_override if _calc_override is not None else ThermodynamicCalculator ()
        p_p_arr =np .concatenate (all_p_p ,axis =0 )
        t_p_arr =np .concatenate (all_t_p ,axis =0 )
        p_e_arr =np .concatenate (all_p_e ,axis =0 )
        t_e_arr =np .concatenate (all_t_e ,axis =0 )

        cv_preds ,cv_tgts =[],[]
        debye_preds ,debye_tgts =[],[]
        if skip_thermo :
            n =len (p_p_arr )
            cv_preds ,cv_tgts =[float ("nan")]*n ,[float ("nan")]*n 
            debye_preds ,debye_tgts =[float ("nan")]*n ,[float ("nan")]*n 
        else :
            for i in range (len (p_p_arr )):
                d_p =calc .compute_Debye_T (p_p_arr [i ])
                d_t =calc .compute_Debye_T (t_p_arr [i ])
                debye_preds .append (d_p )
                debye_tgts .append (d_t )
                cv_p =calc .compute_Cv (p_p_arr [i ],T_range =[300.0 ])[0 ]
                cv_t =calc .compute_Cv (t_p_arr [i ],T_range =[300.0 ])[0 ]
                cv_preds .append (cv_p )
                cv_tgts .append (cv_t )

        df ['cv_pred']=cv_preds 
        df ['cv_true']=cv_tgts 
        df ['debye_pred']=debye_preds 
        df ['debye_true']=debye_tgts 

        valid_debye =(np .array (debye_tgts )>50 )&(np .array (debye_preds )>50 )
        if np .sum (valid_debye )>0 :
            cv_mae =float (np .mean (np .abs (np .array (cv_preds )[valid_debye ]-np .array (cv_tgts )[valid_debye ])))
        elif np .all (~np .isfinite (np .array (cv_preds ,dtype =float ))):
            cv_mae =float ("nan")# thermo skipped (nonuniform grid)
        else :
            cv_mae =float (np .mean (np .abs (np .array (cv_preds )-np .array (cv_tgts ))))
        summary ['cv_mae']=cv_mae 
        summary ['sample_df']=df 
        summary ['pred_phdos']=p_p_arr 
        summary ['pred_edos']=p_e_arr 
        summary ['tgt_phdos']=t_p_arr 
        summary ['tgt_edos']=t_e_arr 

    return summary 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uniARPAT Ablation Experiments Runner')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5', 'all'], help='Model variant to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--skip_existing', action='store_true', help='Skip variant if test summary already exists')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (H1 hygiene, recorded in history CSV)')
    parser.add_argument('--tag', type=str, default='', help='Run tag, e.g. _b96: isolates save_dir/results from h1 outputs')
    parser.add_argument('--data_dir', type=str, default='./data/train4ARPAT', help='Dataset root (C2b: per-arm dir)')
    parser.add_argument('--edos_num', type=int, default=128, help='eDOS output bins (C2b grid arms)')
    parser.add_argument('--phdos_num', type=int, default=64, help='phDOS output bins (C2b grid arms)')
    parser.add_argument('--atom_feat', type=str, default='legacy3', choices=['legacy3', 'mendeleev24'], help='Atom feature table (C1.1)')
    parser.add_argument('--energy_code', type=str, default='none', choices=['none', 'edos'], help='eDOS bin-energy code (C1.2)')
    parser.add_argument('--edos_grid', type=str, default='', help='C1.2 grid key (E0/E2..) in grids.json or path to centers npy')
    parser.add_argument('--tv_w', type=float, default=0.0, help='C1.3 TV weight')
    parser.add_argument('--grad_w', type=float, default=0.0, help='C1.3 gradient-match weight')
    parser.add_argument('--peak_w', type=float, default=1.0, help='C1.3 peak-region weight')
    parser.add_argument('--tail_w', type=float, default=1.0, help='C1.3 phDOS tail weight')
    parser.add_argument('--tail_start', type=int, default=-1, help='C1.3 tail start bin (-1=off)')
    parser.add_argument('--augment', action='store_true', help='C1.4 phonon displacement aug (train only)')
    parser.add_argument('--disp_sigma', type=float, default=0.01, help='C1.4 displacement sigma (frac)')
    parser.add_argument('--norm', type=str, default='sumnorm', choices=['minmax', 'sumnorm'], help='Target norm (C2.1 merged default; minmax recovers legacy)')
    parser.add_argument('--use_mask', action='store_true', help='C2.3 coverage-mask the loss (eval protocol unchanged)')
    parser.add_argument('--dropout', type=float, default=None, help='B4 transformer dropout (default config 0.1)')
    parser.add_argument('--weight_decay', type=float, default=None, help='B4 AdamW weight decay (default 0.01)')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='B4 warmup epochs (default 5)')
    parser.add_argument('--lambda_ph', type=float, default=None, help='B4 phonon loss weight (default 1.0)')
    parser.add_argument('--grad_clip', type=float, default=None, help='B4 grad clip max-norm (default off)')
    parser.add_argument('--w_w1', type=float, default=None, help='L3 W1/CDF term weight (default 1.0)')
    parser.add_argument('--w_huber', type=float, default=None, help='L3 Huber term weight (default 1.0)')
    parser.add_argument('--scale_mode', type=str, default='eta', choices=['none', 'decoupled', 'eta'], help='C2.4/H1 supervised scale/coverage head')
    parser.add_argument('--eta_sup_w', type=float, default=1.0, help='H1 eta/gamma supervision weight')
    parser.add_argument('--delta_edos', type=float, default=0.09375, help='H1 eDOS bin width (E0)')
    parser.add_argument('--delta_phdos', type=float, default=19.6875, help='H1 phDOS bin width (P0)')
    parser.add_argument('--scalar_mode', type=str, default='none', choices=['none', 's1'], help='S1 boundary scalar heads')
    parser.add_argument('--scalar_sup_w', type=float, default=1.0, help='S1 scalar supervision weight')
    parser.add_argument('--use_g1', action='store_true', help='E9-P0 G1 exact sparse graph + hub token')
    parser.add_argument('--g1_r_cut', type=float, default=5.5, help='G1 cutoff Angstrom (Design-E: 5.5)')
    parser.add_argument('--g1_max_neighbors', type=int, default=48, help='G1 per-atom neighbor cap (Design-E: 48)')
    parser.add_argument('--q1_coord', action='store_true', help='E9-P0 Q1 coordinate trunk MLPs')
    parser.add_argument('--q1_hidden', type=int, default=128, help='Q1 trunk hidden dim')
    parser.add_argument('--q2_fourier', action='store_true', help='E9-P0 Q2 RFF trunk (implies trunk pathway)')
    parser.add_argument('--freeze_backbone', action='store_true', help='C2.4 Phase A: train scale head only')
    parser.add_argument('--init_ckpt', type=str, default='', help='C2.4 init weights (strict=False)')
    parser.add_argument('--scale_sup_w', type=float, default=1.0, help='C2.4 scale supervision weight')
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
            args.model = m
            cfg = ExperimentConfig.from_args(args)
            train_and_eval(cfg)
    else:
        cfg = ExperimentConfig.from_args(args)
        train_and_eval(cfg)
