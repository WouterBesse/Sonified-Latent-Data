from tqdm.auto import tqdm
import torch
import os
import math
import argparse
from WaveVae import WaveNetVAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from WVData import WVDataset
import warnings
from datetime import datetime

def calculate_loss(output, target, mu, logvar, kl_term, loss_fn):
    # target = target = torch.unsqueeze(torch.unsqueeze(target[:, -1], 1), 1)
    # print(output[:, -1:, :].size(), 
    reconstruction_loss = loss_fn(output[:, -1, :] * 100, target[:, -1])
    # reconstruction_loss *= math.log2(math.e)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.mean(kl_loss, dim = 0)
    
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

    return max(kl_term, kl_max)

def export_model(model, path, epoch, name = None):
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
    modelname = ''
    if name is None:
        date = datetime.today().strftime('%Y-%m-%d')
        modelname = date + epoch
    else:
        modelname = epoch
    path = os.path.join(path, modelname)
    torch.save(model.state_dict(), path)
    
def validate(model, dataloader, kl_mult, loss_fn, device='cuda', verbose = False):
    model.eval()
    total_eval_loss = [0, 0, 0]
    eval_step = 1
    
    with torch.no_grad():

        with tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating", colour='orange') as t:
            for batch_idx, (onehot_input, mfcc_input, target) in t:
                onehot_input = onehot_input.to(device)
                mfcc_input = mfcc_input.to(device)
                target = target.to(device)

                output, mean, variance = model(onehot_input, mfcc_input, False, verbose)
                real_loss, rec_loss, kl_loss = calculate_loss(
                    output.transpose(1, 2), target, mean, variance, kl_mult, loss_fn)
                total_eval_loss = [
                    total_eval_loss[0] + real_loss.item(),
                    total_eval_loss[1] + rec_loss.item(),
                    total_eval_loss[2] + kl_loss.item()
                ]

                t.set_description(
                    f"Validating. Rec loss: {round(rec_loss.item(), 2)}.")
                eval_step += 1

    return total_eval_loss[0] / eval_step, total_eval_loss[1] / eval_step, total_eval_loss[2] / eval_step


def train(model, dataloader_train, dataloader_val, writer, export_path, learning_rate=0.00001, epoch_amount=100, logs_per_epoch=5, kl_anneal=0.01, max_kl=0.5, device='cuda', verbose = False):
    torch.cuda.empty_cache()
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logstep = 0
    kl_mult = 0.0
    total_step = 0
    export_model(model, export_path, 0)

    for epoch in range(epoch_amount):
        model.train(True)
        total_epoch_loss = [0, 0, 0]
        step = 1
        divstep = 1

        with tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Training. Epoch: {epoch}. Loss for step {step}: n.v.t.", colour='magenta') as t:
            for batch_idx, (onehot_input, mfcc_input, target) in t:
                model.train(True)
                optimizer.zero_grad(set_to_none=True)

                onehot_input = onehot_input.to(device)
                mfcc_input = mfcc_input.to(device)
                target = target.to(device)

                output, mean, variance = model(onehot_input, mfcc_input, True, verbose)

                real_loss, rec_loss, kl_loss = calculate_loss(
                    output.transpose(1, 2), target, mean, variance, kl_mult, loss_fn)
                
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
                writer.add_scalar('Train step loss:',
                                  real_loss.item(), total_step)
                step += 1
                total_step += 1
                divstep += 1

                if step % (len(dataloader_train) // logs_per_epoch) == 0 or step - 1 == 0:

                    eval_loss_real, eval_loss_rec, eval_loss_kl = validate(
                        model, dataloader_val, kl_mult, loss_fn, device, verbose)

                    writer.add_scalars('Validation Loss', {
                        'Real loss': eval_loss_real,
                        'Reconstruction loss': eval_loss_rec,
                        'KL loss': eval_loss_kl
                    }, logstep)

                    writer.add_scalars('Train loss', {
                        'Real loss': total_epoch_loss[0] / divstep,
                        'Reconstruction loss': total_epoch_loss[1] / divstep,
                        'Kl loss': total_epoch_loss[2] / divstep
                    }, logstep)

                    logstep += 1
                    total_epoch_loss = [0, 0, 0]
                    divstep = 0
                    kl_mult = anneal_kl(kl_mult, kl_anneal, max_kl)

        if epoch % 5 == 0:
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
    parser.add_argument('-bs', '--batch_size', help='enter the batch size', type=int, default = 2)
    parser.add_argument('-lr', '--learning_rate', help='enter the learning rate', type=float, default = 0.00001)
    parser.add_argument('-kla', '--kl_anneal', help='enter the kl anneal', type=float, default = 0.01)
    parser.add_argument('-mkl', '--max_kl', help='enter the max kl', type=float, default = 0.5)
    parser.add_argument('-lpe', '--logs_per_epoch', help='enter the logs per epoch', type=int, default = 6)
    parser.add_argument('-d', '--device', help='enter the device', type=str, default = 'cuda:2')
    parser.add_argument('-mf', '--max_files', help='enter the max files', type=int, default = 800)
    

    args = parser.parse_args()

    batchsize = args.batch_size
    device = args.device
    input_size = (40, 112)
    upsamples = [2, 2, 2, 2, 2, 2, 2, 2]
    zsize = 32

    isExist = os.path.exists(args.export_path)
    if not isExist:
        os.makedirs(args.export_path)
        print("Export path didn't exist, created directory")
    else:
        print("Export path exists")

    WaveVAE = WaveNetVAE(input_size,
                        num_hiddens = 768,
                        upsamples = upsamples,
                        zsize = zsize,
                        out_channels = 256)

    WaveVAE.to(device)

    VAEDataset = WVDataset(audio_path = args.train_path,
                        length = 4096,
                        skip_size = 4096 // 2,
                        sample_rate = 24000,
                        max_files = args.max_files,
                        hop_length = 128)

    val_VAEDataset = WVDataset(audio_path = args.validation_path,
                        length = 4096,
                        skip_size = 4096 // 2,
                        sample_rate = 24000,
                        max_files = 200,
                        hop_length = 128)

    VAEDataloader = DataLoader(VAEDataset,
                            batch_size = batchsize,
                            shuffle = True)

    val_VAEDataloader = DataLoader(val_VAEDataset,
                            batch_size = batchsize,
                            shuffle = False)
    
    writer = SummaryWriter()

    train(WaveVAE, VAEDataloader, val_VAEDataloader, 
          writer=writer, 
          export_path = args.export_path,
          learning_rate=args.learning_rate, 
          epoch_amount=args.epochs, 
          logs_per_epoch=args.logs_per_epoch, 
          kl_anneal=args.kl_anneal, 
          max_kl=args.max_kl, 
          device=device)