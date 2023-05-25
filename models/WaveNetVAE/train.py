from tqdm.auto import tqdm
import torch
import os
import math
import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.WaveNetVAE.WaveVae import WaveNetVAE
from models.WaveNetVAE.WVData import WVDataset
# from WaveVae import WaveNetVAE
# from WVData import WVDataset
import warnings
from datetime import datetime

def calculate_loss(output, target, mu, logvar, kl_term, loss_fn):
    # target = target = torch.unsqueeze(torch.unsqueeze(target[:, -1], 1), 1)
    # print(output[:, -1:, :].size(), 
    reconstruction_loss = loss_fn(output[..., -1], target)
    # reconstruction_loss *= math.log2(math.e)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.mean(kl_loss, dim = 0)
    
    return reconstruction_loss + kl_loss * kl_term, reconstruction_loss, kl_loss

# def calculate_loss_new(quant_pred, target_wav, mu, logvar, kl_term, logsoftmax):
#     '''
#     Compute SGVB estimator from equation 8 in
#     https://arxiv.org/pdf/1312.6114.pdf
#     Uses formulas from "Autoencoding Variational Bayes",
#     Appendix B, "Solution of -D_KL(q_phi(z) || p_theta(z)), Gaussian Case"
#     '''
#     # B, T, Q, L: n_batch, n_timesteps, n_quant, n_samples_per_datapoint
#     # K: n_bottleneck_channels
#     # log_pred: (L * B, T, Q), the companded, quantized waveforms.
#     # target_wav: (B, T)
#     # mu, log_sigma_sq: (B, T, K), the vectors output by the bottleneck
#     # Output: scalar, L(theta, phi, x)
#     # log_sigma_sq = self.bottleneck.log_sigma_sq
#     # print('quant_pred size: ', quant_pred.size(), 'target_wav size: ', target_wav.size())
#     log_pred = logsoftmax(quant_pred[...,:-1])
#     # print('log pred size: ', log_pred.size())
#     sigma_sq = logvar
#     mu = mu
#     log_sigma_sq = torch.log(sigma_sq)
#     mu_sq = mu * mu

#     # neg_kl_div_gaussian: (B, K) (from Appendix B at end of derivation)
#     channel_terms = 1.0 + log_sigma_sq -  mu_sq - sigma_sq 
#     neg_kl_div_gauss = 0.5 * torch.sum(channel_terms)

#     L = 1
#     BL = log_pred.size(0)
#     assert BL % L == 0 

#     target_wav_aug = target_wav.repeat(L, 1).unsqueeze(1).long()
#     log_pred_target = torch.gather(log_pred, 1, target_wav_aug)
#     log_pred_target_avg = torch.mean(log_pred_target)

#     log_pred_loss = - log_pred_target_avg
#     kl_div_loss = - neg_kl_div_gauss 

#     # "For the VAE, this collapse can be prevented by annealing the weight
#     # of the KL term and using the free-information formulation in Eq. (2)"
#     # (See p 3 Section C second paragraph)
#     total_loss = (
#         log_pred_loss + kl_term
#         * torch.clamp(kl_div_loss, min=9))

#     return total_loss, log_pred_loss, kl_div_loss


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
        modelname = date + str(epoch)
    else:
        modelname = str(epoch)
    path = os.path.join(path, modelname)
    torch.save(model.state_dict(), path)
    
def validate(model, waveform, mfcc_input, kl_mult, loss_fn, total_eval_loss, t, device='cuda', verbose = False):
    onehot_input = waveform[:, :4096].to(device).unsqueeze(1)
    mfcc_input = mfcc_input.to(device)
    target = waveform.type(torch.LongTensor).to(device)

    output, mean, variance = model(onehot_input, mfcc_input, False, verbose)
    real_loss, rec_loss, kl_loss = calculate_loss(
        output, target[...,-1], mean, variance, kl_mult, loss_fn)
    total_eval_loss = [
        total_eval_loss[0] + real_loss.item(),
        total_eval_loss[1] + rec_loss.item(),
        total_eval_loss[2] + kl_loss.item()
    ]

    t.set_description(
        f"Validating. Rec loss: {round(rec_loss.item(), 2)}.")

    return total_eval_loss

def trainstep(model, waveform, mfcc_input, optimizer, step, total_step, writer, kl_mult, loss_fn, total_epoch_loss, t, verbose, device='cuda'):
    model.train(True)
    optimizer.zero_grad(set_to_none=True)

    onehot_input = waveform[:, :4096].to(device).unsqueeze(1)
    mfcc_input = mfcc_input.to(device)
    target = waveform.type(torch.LongTensor).to(device)

    output, mean, variance = model(onehot_input, mfcc_input, True, verbose)

    real_loss, rec_loss, kl_loss = calculate_loss(
        output, target[...,-1], mean, variance, kl_mult, loss_fn)

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
    
    return total_epoch_loss

def train(model, dataloader_train, writer, export_path, learning_rate=0.00001, epoch_amount=100, logs_per_epoch=5, kl_anneal=0.01, max_kl=0.5, device='cuda', verbose = False):
    torch.cuda.empty_cache()
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.LogSoftmax(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logstep = 0
    kl_mult = 0.001
    total_step = 0
    

    for epoch in range(epoch_amount):
        model.train(True)
        total_epoch_loss = [0, 0, 0]
        total_eval_loss = [0, 0, 0]
        train_step = 1
        eval_step = 1
        divstep = 1
        new_total_step = 1

        with tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc=f"Training. Epoch: {epoch}. Loss for step {train_step}: n.v.t.", colour='magenta') as t:
            for batch_idx, (waveform, mfcc_input) in t:
                if new_total_step % 200 > 30:
                    total_epoch_loss = trainstep(model, waveform, mfcc_input, optimizer, train_step, total_step, writer, kl_mult, loss_fn, total_epoch_loss, t, verbose, device)
                    train_step += 1
                    total_step += 1
                    divstep += 1
                else:
                    model.eval()
                    with torch.no_grad():          
                        total_eval_loss = validate(model, waveform, mfcc_input, kl_mult, loss_fn, total_eval_loss, t, device, verbose)
                        eval_step += 1
                    if new_total_step % 200 == 30:
                        writer.add_scalars('Validation Loss', {
                            'Real loss': total_eval_loss[0] / eval_step,
                            'Reconstruction loss': total_eval_loss[1] / eval_step,
                            'KL loss': total_eval_loss[2] / eval_step
                        }, logstep)

                        writer.add_scalars('Train loss', {
                            'Real loss': total_epoch_loss[0] / divstep,
                            'Reconstruction loss': total_epoch_loss[1] / divstep,
                            'Kl loss': total_epoch_loss[2] / divstep
                        }, logstep)

                        logstep += 1
                        total_epoch_loss = [0, 0, 0]
                        total_eval_loss = [0, 0, 0]
                        divstep = 0
                        eval_step = 1
                        kl_mult = anneal_kl(kl_mult, kl_anneal, max_kl)
                        
                new_total_step += 1

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
    input_size = (60, 112)
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
    export_model(WaveVAE, args.export_path, 0)

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