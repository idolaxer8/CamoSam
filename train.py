import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from model_single import OurBigModelEmb
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from Data_loading import get_loader, get_test_loader
import torchvision.transforms as T
import matplotlib.pyplot as plt  # Import Matplotlib
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard SummaryWriter
import torchvision.utils as vutils


# Define the transform to convert tensors to PIL images
to_pil = T.ToPILImage()


# Accuracy parameters
FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
M = MAE()

# Set Parametets
batch_size = 3
train_size = 416
test_size = 416
NUM_EPOCHS = 10

def save_img(orig_img, mask, gt, save_dir, epoch, ix, i, writer=None):
    # Convert to PIL images for saving
    orig_img_pil = to_pil(orig_img)
    mask_pil = to_pil(mask.squeeze(0))
    gt_pil = to_pil(gt.squeeze(0))

    # Save images
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(orig_img_pil)
    ax[0].set_title('Input')
    ax[1].imshow(mask_pil)
    ax[1].set_title('Output')
    ax[2].imshow(gt_pil)
    ax[2].set_title('Ground Truth')
    plt.tight_layout()
    image_path = os.path.join(save_dir, f'epoch_{epoch}_batch{ix}_img{i}.png')
    plt.savefig(image_path)
    plt.close()

    # Log images to TensorBoard
    if writer is not None:
        # Stack images side by side
        if mask.size()[1] != 416: 
            resize_T = T.Resize((416,416))
            mask = resize_T(mask)
        # Ensure masks and gt have 3 channels
        mask_3ch = mask.repeat(3, 1, 1)  # Repeat along the channel dimension
        gt_3ch = gt.repeat(3, 1, 1)      # Repeat along the channel dimension
        img_grid = vutils.make_grid([orig_img, gt_3ch, mask_3ch])
        writer.add_image(f'Epoch_{epoch}/Images_batch{ix}_img{i}', img_grid, epoch)


def loss_graph_plot(loss_list, num_epochs):
    epochs_list = list(range(num_epochs))
    plt.plot(epochs_list, loss_list, label='DICE and BCE loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Dice and BCE loss as a function of epochs", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts, size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam, optimizer, epoch, writer):
    save_dir = os.path.join('runs_train', f'matplotlib_{args["folder"]}_epoch{epoch}')
    os.makedirs(save_dir, exist_ok=True)    
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    optimizer.zero_grad()

    # Define the transform to convert tensors to PIL images
    to_pil = T.ToPILImage()

    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Camouflaged',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))

        # Log images to TensorBoard every 50 batches
        if ix % 50 == 0:
            for i in range(len(orig_imgs)):
                orig_img = orig_imgs[i].cpu()
                mask = masks[i].cpu()
                gt = gts[i].cpu()
                save_img(orig_img, mask, gt, save_dir, epoch, ix, i, writer=writer)

    return np.mean(loss_list)


def valid_single_epoch(ds, model, sam, epoch, args, writer):
    save_dir = os.path.join('runs_valid', f'matplotlib_{args["folder"]}_epoch{epoch}')
    os.makedirs(save_dir, exist_ok=True) 
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts, input_size=input_size, original_size=original_size)
        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        # Convert tensors to numpy arrays
        masks_np = masks.squeeze().detach().cpu().numpy()
        gts_np = gts.squeeze().detach().cpu().numpy()

        dice, ji = get_dice_ji(masks_np, gts_np)
        FM.step(pred=masks_np, gt=gts_np)
        WFM.step(pred=masks_np, gt=gts_np)
        SM.step(pred=masks_np, gt=gts_np)
        EM.step(pred=masks_np, gt=gts_np)
        M.step(pred=masks_np, gt=gts_np)
        iou_list.append(ji)
        dice_list.append(dice)

        fm_results = FM.get_results()
        wfm_results = WFM.get_results()
        sm_results = SM.get_results()
        em_results = EM.get_results()
        m_results = M.get_results()

        # Extract individual values
        fm_val = fm_results["fm"]
        wfm_val = wfm_results["wfm"]
        sm_val = sm_results["sm"]
        em_val = em_results["em"]
        m_val = m_results["mae"]
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f} :: FM: {fm:.4f} :: WFM: {wfm:.4f} :: SM: {sm:.4f} :: EM: {em:.4f} :: M: {m:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list),
                fm=fm_val['adp'],
                wfm=wfm_val,
                sm=sm_val,
                em=em_val['adp'],
                m=m_val
            ))
        # Log images to TensorBoard every 50 batches
        if ix % 50 == 0:
            for i in range(len(orig_imgs)):
                orig_img = orig_imgs[i].cpu()
                mask = masks[i].cpu()
                gt = gts[i].cpu()
                save_img(orig_img, mask, gt, save_dir, epoch, ix, i, writer=writer)


    model.train()
    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def main(args=None, sam_args=None):
    # Gpu colab check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training with {str(device).upper()}')

    # Initialize TensorBoard writer
    train_writer = SummaryWriter(log_dir=os.path.join('runs_train', f'tensorboard_{args["folder"]}'))
    valid_writer = SummaryWriter(log_dir=os.path.join('runs_valid', f'tensorboard_{args["folder"]}'))
    model = OurBigModelEmb(args=args).to(device)
    # SAM Initialize
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD']))
    # Path definition
    img_root = "Data/TrainDataset/TrainDataset/Imgs/"
    gt_root = "Data/TrainDataset/TrainDataset/GT/"
    edge_root = "Data/TrainDataset/TrainDataset/Edge/"
    img_root_t = "Data/TestDataset/TestDataset/COD10K/Imgs/"
    gt_root_t = "Data/TestDataset/TestDataset/COD10K/GT/"
   
    # DATA Loading
    train_data = get_loader(img_root, gt_root, edge_root, batch_size, train_size)
    valid_data = get_test_loader(img_root_t, gt_root_t, test_size)
    
    best = 0
    train_loss = []
    path_best = 'results/gpu' + str(args['folder']) + '/best.csv'
    f_best = open(path_best, 'w')
    for epoch in range(NUM_EPOCHS):
        train_loss.append(train_single_epoch(train_data, model.train(), sam.eval(), optimizer, epoch, train_writer))
        with torch.no_grad():
            IoU_val = valid_single_epoch(valid_data, model.eval(), sam, epoch, args, valid_writer)
            if IoU_val > best:
                torch.save(model, args['path_best'])
                best = IoU_val
                print('best results: ' + str(best))
                if (best is not None) and (f_best is not None):
                    f_best.write(str(epoch) + ',' + str(best) + '\n')
                    f_best.flush()
    
    
    loss_graph_plot(train_loss, NUM_EPOCHS)
    train_writer.close()
    valid_writer.close()
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-Idim', '--Idim', default=416, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['path'] = os.path.join('results',
                                'gpu' + folder,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    os.mkdir(args['vis_folder'])
    sam_args = {
        'sam_checkpoint': "code/segment_anything/sam_vit_h.pth",
        'model_type': "vit_h",
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    main(args=args, sam_args=sam_args)
