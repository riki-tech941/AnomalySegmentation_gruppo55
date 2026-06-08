import os
import sys
import cv2
import glob
import importlib
import torch
import random
import yaml
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models.eomt import EoMT
from models.vit import ViT
from torchvision.transforms.functional import to_tensor, pil_to_tensor
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast

device = "cuda"
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Function infer_semantics modified to only get the logits

def infer_semantic(img, model, args):
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = model.window_imgs_semantic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(

            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

    return logits





def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir',default="./")
    parser.add_argument('--loadWeights')
    parser.add_argument('--configPath')
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="./anomaly/Anomaly_Datasets")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='maxlogit', choices=['maxlogit', 'msp', 'entropy','rba','temperature'], help='Metodo per calcolare anomaly score')
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    weightspath = args.loadDir + args.loadWeights
    print ("Loading weights: " + weightspath)

    with open(args.configPath, "r") as f:
      config = yaml.safe_load(f)

    if "num_classes" in config.get("data", {}) :
       NUM_CLASSES = config["data"]["num_classes"]
    if "num_classes" in config.get("model", {}).get("init_args", {}) :
       NUM_CLASSES = config["model"]["init_args"]["num_classes"]
    else :
       NUM_CLASSES = 19

    if "img_size" in config.get("data", {}) :
       IMG_SIZE = config["data"]["img_size"]
    if "img_size" in config.get("data", {}).get("init_args") :
       IMG_SIZE = config["data"]["init_args"]["img_size"]
    else :
       IMG_SIZE = (1024, 1024)
      
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=IMG_SIZE, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config.get("data", {}).get("init_args", {}):
      model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]
    model_kwargs.pop("num_classes", None)
    
    model = (
        lit_cls(
            img_size=IMG_SIZE,
            num_classes=NUM_CLASSES,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")

    state_dict_path=weightspath

    is_dinov3 = "dinov3" in name

    if is_dinov3:
        model_kwargs["ckpt_path"] = state_dict_path
        model_kwargs["delta_weights"] = True

    if not is_dinov3:
        state_dict = torch.load(
            state_dict_path, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)

    print ("Model and weights LOADED successfully")
    model.eval()

    #Iniziallizzo gli t da testare e i dizinari su cui accumulare i risultati nel caso di temperature
    T = [x / 100 for x in range(10, 300, 50)]
    T.extend([0.5, 0.75, 1.1])  # aggiunge i tre valori da testare per tabella singolarmente
    T = sorted(set(T))          # rimuove eventuali duplicati e ordina
    temp_scores = {key : [] for key in T}
    temp_gts = {key : [] for key in T}
    temp_aupcr = {key : None for key in T}
    temp_fpr95 = {key : None for key in T}
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        #Usiamo pil_to_tensor e non to tensor (che nel file originale era contenuto in input_trasform) perche ci
        #mantiene in formato uint che ci serve per la funzione window_imgs_semantic all interno di infer_semantic
        images = pil_to_tensor(Image.open(path).convert('RGB')).cuda() # uint8, shape [3, H, W], valori 0-255
        with torch.no_grad():
            result = infer_semantic(images, model, args)
        #anomaly_result = 1.0 - np.max(result[0].data.cpu().numpy(), axis=0)
        #Inizio modifica, implementiamo scelta tra i metodi di valutazione:
        if args.method == 'maxlogit':
          anomaly_result = 1.0 - np.max(result[0].data.cpu().numpy(), axis=0)
        elif args.method == 'msp':
          # Applichiamo softmax per convertire i logit in probabilità
          probs = torch.softmax(result[0], dim=0)
          anomaly_result = 1.0 - np.max(probs.data.cpu().numpy(), axis=0)
        elif args.method == 'entropy':
            probs = torch.softmax(result[0], dim = 0)
            # Aggiungiamo 1e-8 a probs dentro il logaritmo per evitare l'errore log(0) = NaN???
            entropy = torch.sum(-probs * torch.log(probs), dim=0)
            # Normalizziamo dividendo per log(numero_classi)
            entropy = entropy / torch.log(torch.tensor(probs.shape[0], dtype=torch.float32))
            anomaly_result = entropy.squeeze(0).data.cpu().numpy()
        elif args.method == 'rba':
            anomaly_result = -result[0].tanh().sum(dim=0).data.cpu().numpy()

        elif args.method == 'temperature':
          for t in T:
            result_auprc = []
            result_fpr = []
            result_t = result[0]/t
            probs_t = torch.softmax(result_t, dim=0)
            anomaly_result_t = 1.0 - np.max(probs_t.data.cpu().numpy(), axis=0)

            pathGT_t = path.replace("images", "labels_masks")
            if "RoadObsticle21" in pathGT_t:
                pathGT_t = pathGT_t.replace("webp", "png")
            if "fs_static" in pathGT_t:
                pathGT_t = pathGT_t.replace("jpg", "png")
            if "RoadAnomaly" in pathGT_t:
                pathGT_t = pathGT_t.replace("jpg", "png")

            mask_t = Image.open(pathGT_t)
            ood_gts_t = np.array(mask_t)

            if "RoadAnomaly" in pathGT_t:
                ood_gts_t = np.where((ood_gts_t == 2), 1, ood_gts_t)
            if "LostAndFound" in pathGT_t:
                ood_gts_t = np.where((ood_gts_t == 0), 255, ood_gts_t)
                ood_gts_t = np.where((ood_gts_t == 1), 0, ood_gts_t)
                ood_gts_t = np.where((ood_gts_t > 1) & (ood_gts_t < 201), 1, ood_gts_t)
            if "Streethazard" in pathGT_t:
                ood_gts_t = np.where((ood_gts_t == 14), 255, ood_gts_t)
                ood_gts_t = np.where((ood_gts_t < 20), 0, ood_gts_t)
                ood_gts_t = np.where((ood_gts_t == 255), 1, ood_gts_t)

            if 1 not in np.unique(ood_gts_t):
                continue

            # Accumula invece di calcolare subito
            temp_scores[t].append(anomaly_result_t)
            temp_gts[t].append(ood_gts_t)

            del result_t, anomaly_result_t, ood_gts_t, mask_t
            torch.cuda.empty_cache()
          continue  # salta il resto del loop

        if args.method in ['maxlogit', 'msp', 'entropy','rba']:
          pathGT = path.replace("images", "labels_masks")
          if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
          if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
          if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

          mask = Image.open(pathGT)

          ood_gts = np.array(mask)

          if "RoadAnomaly" in pathGT:
              ood_gts = np.where((ood_gts==2), 1, ood_gts)
          if "LostAndFound" in pathGT:
              ood_gts = np.where((ood_gts==0), 255, ood_gts)
              ood_gts = np.where((ood_gts==1), 0, ood_gts)
              ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

          if "Streethazard" in pathGT:
              ood_gts = np.where((ood_gts==14), 255, ood_gts)
              ood_gts = np.where((ood_gts<20), 0, ood_gts)
              ood_gts = np.where((ood_gts==255), 1, ood_gts)

          if 1 not in np.unique(ood_gts):
              continue
          else:
              ood_gts_list.append(ood_gts)
              anomaly_score_list.append(anomaly_result)
          del result, anomaly_result, ood_gts, mask
          torch.cuda.empty_cache()

    #file.write( "\n")

    if args.method == 'temperature':
      for t in T:
        ood_gts = np.array(temp_gts[t])
        anomaly_scores = np.array(temp_scores[t])
        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)
        ood_out = anomaly_scores[ood_mask]
        ind_out = anomaly_scores[ind_mask]
        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))
        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))
        prc_auc = average_precision_score(val_label, val_out)
        fpr = fpr_at_95_tpr(val_out, val_label)
        print(f'[Temperature={t}] AUPRC: {prc_auc * 100.0:.2f}%  |  FPR@TPR95: {fpr * 100.0:.2f}%')
        temp_aupcr[t] = prc_auc
        temp_fpr95[t] = fpr
        #file.write(f'\n[Temperature={t}] AUPRC: {prc_auc * 100.0:.2f}%  |  FPR@TPR95: {fpr * 100.0:.2f}%')

      best_t_auprc = max(temp_aupcr, key=temp_aupcr.get)
      best_t_fpr = min(temp_fpr95, key=temp_fpr95.get)

      print(f'Miglior AUPRC: {temp_aupcr[best_t_auprc]*100.0:.2f}% con T={best_t_auprc}')
      print(f'Miglior FPR@TPR95: {temp_fpr95[best_t_fpr]*100.0:.2f}% con T={best_t_fpr}')

    else:

      ood_gts = np.array(ood_gts_list)
      anomaly_scores = np.array(anomaly_score_list)

      ood_mask = (ood_gts == 1)
      ind_mask = (ood_gts == 0)

      ood_out = anomaly_scores[ood_mask]
      ind_out = anomaly_scores[ind_mask]

      ood_label = np.ones(len(ood_out))
      ind_label = np.zeros(len(ind_out))

      val_out = np.concatenate((ind_out, ood_out))
      val_label = np.concatenate((ind_label, ood_label))

      prc_auc = average_precision_score(val_label, val_out)
      fpr = fpr_at_95_tpr(val_out, val_label)

      print(f'AUPRC score: {prc_auc*100.0}')
      print(f'FPR@TPR95: {fpr*100.0}')


    #file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))








if __name__ == '__main__':
    main()