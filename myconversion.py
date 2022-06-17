"""
Generate speaker embeddings and metadata for training
"""
import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
from model_bl import D_VECTOR
from collections import OrderedDict
import torch
import numpy as np
from math import ceil
from model_vc import Generator
import soundfile as sf
from my_synthesis import build_model
from my_synthesis import wavegen
from my_plot import *
import getopt,sys
from inspect import currentframe, getframeinfo

inputfile = ''
outputfile = ''
try:
    help_tip='xxx.py -i <input wave> -r <target wav>'
    opts, args = getopt.getopt(sys.argv[1:],"hi:r:")
except getopt.GetoptError as err:
    print(err)
    print(help_tip,getframeinfo(currentframe()).lineno)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_tip,getframeinfo(currentframe()).lineno)
        sys.exit()
    elif opt in ("-i"):
        sourcefile = arg
    elif opt in ("-r"):
        referencefile = arg

outputfile= sourcefile.split('.')[0] +'2'+ referencefile
print ('源文件为：', sourcefile)
print ('参考风格文件为：', referencefile)
print ('输出文件为：', outputfile)
# sys.exit(0)

# %% 加载模型
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
num_uttrs = 10
len_crop = 128

device = 'cuda:0'
G = Generator(32,256,512,32).eval().to(device)
g_checkpoint = torch.load('autovc.ckpt',device)
G.load_state_dict(g_checkpoint['model'])

#vocoder
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])



def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)
def get_mel_and_emb(filename):
    x, fs = sf.read(filename)
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    wav=y
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = np.clip((D_db + 100) / 100, 0, 1)

    tmp=S.astype(np.float32)
    melsp0 = torch.from_numpy(tmp[np.newaxis, :, :]).cuda()
    #这里验证过，melsp0的计算和vocoder的合成是没有问题的。
    # waveform = wavegen(model, c=melsp0[0])
    # sf.write('r0.wav', waveform, samplerate=16000)
    embs=[]
    for i in range(10):
        left = np.random.randint(0, tmp.shape[0] - len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left + len_crop, :]).cuda()
        emb = C(melsp)
        embs.append(emb)
    e1=torch.stack(embs)
    e2=torch.mean(e1,dim=0)
    # emb=torch.reshape(emb,(1,-1))
    return melsp0,e2

uttr_org,emb_org=get_mel_and_emb(sourcefile)
_,emb_trg=get_mel_and_emb(referencefile)

def my_pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.size()[1])/base))
    len_pad = len_out - x.size()[1]
    assert len_pad >= 0
    if len_pad>0:
        return torch.cat([x,torch.zeros(1,len_pad,x.size()[2]).to(device)],dim=1)
    else:
        return x;

uttr_org=my_pad_seq(uttr_org)

_, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
spect=x_identic_psnt


#进行语音合成
waveform = wavegen(model, c=spect[0][0])
sf.write(outputfile, waveform, samplerate=16000)

