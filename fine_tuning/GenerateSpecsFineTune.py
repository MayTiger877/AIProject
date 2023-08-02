from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *

########################     spectrogram prep    #######################

def FineTune_reverb_noisy_data():
    rir_dir = "/home/may.tiger/AIProject/data/ClassroomOmni"
    moist_dir = "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/noisy_only"
    wet_dir = "/home/may.tiger/AIProject/big_data_set/fine_tune_data/test/noisy_reverb"

    sys.path.append(moist_dir)
    sys.path.append(rir_dir)

    rir_file_names = []
    for subdir, dirs, files in os.walk(rir_dir):
        for file in files:
            if (".wav" in file):
                rir_file_names.append(os.path.join(subdir,file))

    audio_file_names = []
    for subdir, dirs, files in os.walk(moist_dir):
        for file in files:
            if (".wav" in file):
                audio_file_names.append(os.path.join(subdir,file))

    print ("RIRs found: " + str(len(rir_file_names)))
    print ("Audio files found: " + str(len(audio_file_names)))

    for i in range(0, len(audio_file_names)):
        print("Proccesing audio file n°: " + str(i+1), flush=True)
        
        rir_index = random.sample(range(len(rir_file_names)), 1)[0]
        ir_audio, ir_time, ir_rate = extract_audio(rir_file_names[rir_index])
        moist_audio, speech_time, speech_rate = extract_audio(audio_file_names[i])
        moist_reverb_audio = discrete_conv(moist_audio, ir_audio, speech_rate, ir_rate)
        moist_reverb_audio = moist_reverb_audio[0:len(moist_audio)]
        save_path = wet_dir + '/' + audio_file_names[i].split('/')[-1]
        sf.write(save_path, moist_reverb_audio, 16000)
            
    print('Saved data', flush=True)

def generate_specs(clean_audio_dir, noisy_audio_dir, lower_bound, upper_bound, checkpoints):
    checkpointX = checkpoints[0]
    checkpointY = checkpoints[1]
    checkpoint_wavenoisy = checkpoints[2]
    checkpoint_wavetarget = checkpoints[3]
    
    sys.path.append(clean_audio_dir)
    sys.path.append(noisy_audio_dir)

    clean_audio_file_names = []
    for subdir, dirs, files in os.walk(clean_audio_dir):
        for file in files:
            if (".wav" in file):
                clean_audio_file_names.append(os.path.join(subdir,file))
                
    
    print ("Audio files found: " + str(len(clean_audio_file_names)))
    
    noisy_audio_file_names = []
    for subdir, dirs, files in os.walk(noisy_audio_dir):
        for file in files:
            if (".wav" in file):
                noisy_audio_file_names.append(os.path.join(subdir,file))
    
    print ("Audio files found: " + str(len(noisy_audio_file_names)))
    
    time_size = 340
    frequency_size = 128
    X = torch.zeros((len(clean_audio_file_names)-0, 1, frequency_size, time_size))
    y = torch.zeros((len(clean_audio_file_names)-0, 1, frequency_size, time_size))

    wave_targets = []
    wave_noisy = []

    for i in range(0, len(clean_audio_file_names)):
        clean_speech_audio, clean_speech_time, clean_speech_rate = extract_audio(clean_audio_file_names[i])
        wave_targets.append(clean_speech_audio)
        clean_speech_spec = generate_spec(clean_speech_audio, clean_speech_rate)
        
        noisy_speech_audio, noisy_speech_time, noisy_speech_rate = extract_audio(noisy_audio_file_names[i])
        wave_targets.append(noisy_speech_audio)
        noisy_speech_spec = generate_spec(noisy_speech_audio, noisy_speech_rate)
        
        clean_speech_spec = cv2.resize(clean_speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)
        noisy_speech_spec = cv2.resize(noisy_speech_spec, dsize = (time_size, frequency_size), interpolation = cv2.INTER_LANCZOS4)

        print("Proccesing audio file n°: " + str(i+1), flush = True)
        X[i-0, 0, :, :] = torch.tensor(noisy_speech_spec)
        y[i-0, 0, :, :] = torch.tensor(clean_speech_spec)
            
        if ((i+1)%500 == 0):    
            torch.save(X, checkpointX)
            torch.save(y, checkpointY)
            torch.save(wave_noisy, checkpoint_wavenoisy)
            torch.save(wave_targets, checkpoint_wavetarget)
            print('Saved data', flush = True)
          
        if ((i+1)==(len(noisy_audio_file_names))):
            torch.save(X, checkpointX)
            torch.save(y, checkpointY)
            torch.save(wave_noisy, checkpoint_wavenoisy)
            torch.save(wave_targets, checkpoint_wavetarget)
            print('Saved data', flush = True)
            return X, y

    return X, y

def FineTune_gen_spec():
    dry_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/fine_tune_data/train/clean'
    wet_audio_rootdir = '/home/may.tiger/AIProject/big_data_set/fine_tune_data/train/noisy_reverb'

    checkpointX =           '/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/noisyspecs.pth'
    checkpointY =           '/home/may.tiger/AIProject/fine_tuning/NoisyReverbedSpecs/cleanspecs.pth'
    checkpoint_wavenoisy =  '/home/may.tiger/AIProject/fine_tuning/NoisyReverbWav/wavenoisy.pth'
    checkpoint_wavetarget = '/home/may.tiger/AIProject/fine_tuning/NoisyReverbWav/waveclean.pth'

    checkpoints = [checkpointX, checkpointY, checkpoint_wavenoisy, checkpoint_wavetarget]

    print("starting generating\n")
    dir_len = len([entry for entry in os.listdir(dry_audio_rootdir) if os.path.isfile(os.path.join(dry_audio_rootdir, entry))])
    print(dir_len)
    X, y = generate_specs(dry_audio_rootdir, wet_audio_rootdir, 0, dir_len, checkpoints)

    print("finished generating\n")