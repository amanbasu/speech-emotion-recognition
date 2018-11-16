import glob
from python_speech_features import mfcc,logfbank
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import webrtcvad
import scipy

def get_vector(sig,rate):
    vec=np.empty((1,3))
    start=0
    end=320

    while(sig.shape[0]>=end+160):
        vad = webrtcvad.Vad()
        vad.set_mode(2)
        
        res=vad.is_speech(sig[start:end].tobytes(),rate)               # speech probability
        zero_crosses = np.nonzero(np.diff(sig[start:end] > 0))[0].shape[0]/0.02 # zero crosses
        f=scipy.fft(sig[start:end])
        f0=min(np.absolute(f))                                         # f0 frequency
        
        start=start+160
        end=end+160

        vec=np.vstack((vec,np.array([res,zero_crosses,f0],ndmin=2)))
    
    mfcc_feat=mfcc(sig,rate,numcep=12,winlen=0.020)[0:vec.shape[0],:]  # mfcc 
    fbank=logfbank(sig,rate,nfilt=5)[0:vec.shape[0],:]                 # log filterbank energies 
    mfcc_grad=np.gradient(mfcc_feat,axis=0)                            # mfcc first derivative
    
    final_feature=np.hstack((mfcc_feat,mfcc_grad,fbank,vec))
    
    return final_feature


df=pd.DataFrame()
for i in range(1,6):
    for file in glob.glob("/IEMOCAP_full_release/Session{}/sentences/wav/*/*.wav".format(i)):
        print(file)
        (rate,sig) = wav.read(file)
        
        final_vector=get_vector(sig,rate)
        feed_dict={"Features":final_vector.astype(np.float64),"name":file.split('/')[-1].split('.')[0]}
        df=df.append(feed_dict,ignore_index=True)
    
df.to_pickle("features")


    
