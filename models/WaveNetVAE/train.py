from tqdm.auto import tqdm
import torch
import os
import math
import argparse

from torch.utils.data import DataLoader
from models.WaveNetVAE.WaveVae import WaveNetVAE
from models.WaveNetVAE.WVData import WVDataset
import warnings
from datetime import datetime
import wandb

def calculate_loss(output, target, mu, logvar, kl_term, loss_fn):
    # target = target = torch.unsqueeze(torch.unsqueeze(target[:, -1], 1), 1)
    # print(output[:, -1:, :].size(), 
    reconstruction_loss = loss_fn(output, target)
    # reconstruction_loss *= math.log2(math.e)
    kl_loss = torch.mean(-0.5 * torch.sum(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 2), dim = 1), dim = 0)
    
    return reconstruction_loss + kl_loss * kl_term, reconstruction_loss, kl_loss

def clear_screen():
    # Clearing the Screen
    # posix is os name for Linux or mac
    if (os.name == 'posix'):
        os.system('clear')
        # else screen will be cleared for windows
    else:
        os.system('cls')

def anneal_kl(kl_term, kl_annealing, kl_max):
    kl_term += kl_annealing

    return min(kl_term, kl_max)

def export_model(model, path, epoch, name = None):
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
    modelname = ''
    if name is None:
        date = datetime.today().strftime('%Y-%m-%d')
        modelname = date + str(epoch)
    else:
        modelname = str(epoch)
    path = os.path.join(path, modelname)
    torch.save(model.state_dict(), path)
    
def validate(model, dataloader, kl_mult, loss_fn, device='cuda', verbose = False):
    model.eval()
    total_eval_loss = [0, 0, 0]
    eval_step = 1
    
    with torch.no_grad():

        with tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating", colour='orange') as t:
            for batch_idx, (snippet, mfcc_input) in t:
                snippet = snippet.to(device)
                mfcc_input = mfcc_input.to(device)

                output, mean, variance = model(snippet[...,:4096].unsqueeze(1), mfcc_input, False, verbose)
                real_loss, rec_loss, kl_loss = calculate_loss(
                    output[..., -1], snippet[..., -1].type(torch.LongTensor).to(device), mean, variance, kl_mult, loss_fn)
                total_eval_loss = [
                    total_eval_loss[0] + real_loss.item(),
                    total_eval_loss[1] + rec_loss.item(),
                    total_eval_loss[2] + kl_loss.item()
                ]

                t.set_description(
                    f"Validating. Rec loss: {round(rec_loss.item(), 2)}.")
                eval_step += 1

    return total_eval_loss[0] / eval_step, total_eval_loss[1] / eval_step, total_eval_loss[2] / eval_step


def train(model, dataloader_train, dataloader_val, export_path, learning_rate=0.00001, epoch_amount=100, logs_per_epoch=5, kl_anneal=0.01, max_kl=0.5, device='cuda', verbose = False):    
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    wandb.watch(model, log="all", log_freq=1)
    logstep = 0
    kl_mult = 0.01
    total_step = 0
    

    for epoch in range(epoch_amount):
        model.train(True)
        total_epoch_loss = [0, 0, 0]
        step = 1
        divstep = 1

        with tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Training. Epoch: {epoch}. Loss for step {step}: n.v.t.", colour='magenta') as t:
            for batch_idx, (snippet, mfcc_input) in t:
                model.train(True)
                optimizer.zero_grad(set_to_none=True)

                snippet = snippet.to(device)
                mfcc_input = mfcc_input.to(device)
                # print('snippet size:', snippet[...,:4096].unsqueeze(1).size())
                with torch.cuda.amp.autocast():
                    output, mean, variance = model(snippet[...,:4096].unsqueeze(1), mfcc_input, True, verbose)
                    # print('generated size:', output[..., -1].size(), 'target size: ', snippet[..., -1].size())
                    real_loss, rec_loss, kl_loss = calculate_loss(
                        output[..., -1], snippet[..., -1].type(torch.LongTensor).to(device), mean, variance, kl_mult, loss_fn)
            
                real_loss.backward()
                optimizer.step()

                # Save losses for total, reconstruction and kl seperately for better inspection of optimisation for different parts
                total_epoch_loss = [
                    total_epoch_loss[0] + real_loss.item(),
                    total_epoch_loss[1] + rec_loss.item(),
                    total_epoch_loss[2] + kl_loss.item()
                ]

                t.set_description(
                    f"Training. Rec/real loss for step {step}: {round(rec_loss.item(), 2)}/{round(real_loss.item(), 2)}.")

                step += 1
                total_step += 1
                divstep += 1

                if step % (len(dataloader_train) // logs_per_epoch) == 0 or step - 1 == 0:

                    eval_loss_real, eval_loss_rec, eval_loss_kl = validate(
                        model, dataloader_val, kl_mult, loss_fn, device, verbose)
                    
                    wandb.log({'epoch': epoch,
                                'train_loss_comb': total_epoch_loss[0] / divstep,
                                'train_loss_rec': total_epoch_loss[1] / divstep,
                                'train_loss_kl': total_epoch_loss[2] / divstep,
                                'val_loss_comb': eval_loss_real,
                                'val_loss_rec': eval_loss_rec,
                                'val_loss_kl': eval_loss_kl,
                                'kl_rate': kl_mult})

                    logstep += 1
                    total_epoch_loss = [0, 0, 0]
                    divstep = 0
                    kl_mult = anneal_kl(kl_mult, kl_anneal, max_kl)

        # if epoch % 5 == 0:
        export_model(model, export_path, epoch=epoch)
    
    export_model(model, export_path, epoch='final')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    clear_screen()
    
    parser = argparse.ArgumentParser(description = 'Train the model')
    parser.add_argument('-tp', '--train_path', help='enter the path to the training data', type=str)
    parser.add_argument('-vp', '--validation_path', help='enter the path to the validation data', type=str)
    parser.add_argument('-ep' , '--epochs', help='enter the amount of epochs', type=int)
    parser.add_argument('-ex', '--export_path', help='Location to export models', type=str, default = './exports/')
    parser.add_argument('-bs', '--batch_size', help='enter the batch size', type=int, default = 32)
    parser.add_argument('-lr', '--learning_rate', help='enter the learning rate', type=float, default = 0.00001)
    parser.add_argument('-kla', '--kl_anneal', help='enter the kl anneal', type=float, default = 0.01)
    parser.add_argument('-mkl', '--max_kl', help='enter the max kl', type=float, default = 0.5)
    parser.add_argument('-lpe', '--logs_per_epoch', help='enter the logs per epoch', type=int, default = 6)
    parser.add_argument('-d', '--device', help='enter the device', type=str, default = 'cuda:2')
    parser.add_argument('-mf', '--max_files', help='enter the max files', type=int, default = 800)
    

    args = parser.parse_args()

    batchsize = args.batch_size
    device = args.device
    input_size = (39, 112)
    upsamples = [2, 2, 2, 2, 2, 2, 2, 2]
    zsize = 32

    isExist = os.path.exists(args.export_path)
    if not isExist:
        os.makedirs(args.export_path)
        print("Export path didn't exist, created directory")
    else:
        print("Export path exists")

    device='cuda:4'
    input_size = (39, 112)
    upsamples = [2, 2, 2, 2, 2, 2, 2, 2]
    zsize = 32

    WaveVAE = WaveNetVAE(input_size,
                         num_hiddens = 768,
                         upsamples = upsamples,
                         zsize = zsize,
                         out_channels = 256)

    WaveVAE = WaveVAE.to(device)

    WaveVAE = torch.nn.DataParallel(WaveVAE, device_ids=[4,5,6,7])
    export_model(WaveVAE, args.export_path, 0)

    VAEDataset = WVDataset(audio_path = args.train_path,
                        length = 4096,
                        skip_size = 4096 // 2,
                        sample_rate = 16000,
                        max_files = args.max_files)

    val_VAEDataset = WVDataset(audio_path = args.validation_path,
                        length = 4096,
                        skip_size = 4096 // 2,
                        sample_rate = 16000,
                        max_files = 32)

    VAEDataloader = DataLoader(VAEDataset,
                            batch_size = batchsize,
                            shuffle = True)

    val_VAEDataloader = DataLoader(val_VAEDataset,
                            batch_size = batchsize,
                            shuffle = False)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="WavenetVAE",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.000001,
        "architecture": "WavenetVAE",
        "dataset": "LJSpeech",
        "epochs": 100,
        "max_files_train": 0,
        "max_files_eval": 32,
        "bias": False,
        "init": "Xavier_U",
        "divide skips & res": False,
        "MFCC_Norm": True
        }
    )

    train(WaveVAE, VAEDataloader, val_VAEDataloader,
          export_path = args.export_path,
          learning_rate=args.learning_rate, 
          epoch_amount=args.epochs, 
          logs_per_epoch=args.logs_per_epoch, 
          kl_anneal=args.kl_anneal, 
          max_kl=args.max_kl, 
          device=device)