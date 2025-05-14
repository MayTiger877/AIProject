**Audio enhancment with UNET and LS-UNET**

Implementation was originally done by D. Le´on and F. A. Tobar at https://github.com/DiegoLeon96/Neural-Speech-Dereverberation. Modifications were done to single functions and the pipeline and data handeling.


(Emphasis was on speech so relevant frequency range displayed is 20-8300 Hz)
First pre-training was doen on the original work LS-UNET to clean reverberation from the audio source.
original signal:
![image](https://github.com/user-attachments/assets/59a6d8b3-1611-4c6d-bc8e-4e9284f82dbf)
Reverberated signal:
![image](https://github.com/user-attachments/assets/9c3a6dc7-b493-478e-a781-3b9fac8fea70)
Pre-train results and dereverberated signal:
![image](https://github.com/user-attachments/assets/4d2675a5-6139-4e68-83f4-53e10a9faf26)
![image](https://github.com/user-attachments/assets/233db4fc-6622-4723-8aa7-8e23265b7ab9)


Second pre=training was doen with a regular UNET in order to segment the signal and result in backgroung noise cleaning:
Noisy signal:
![image](https://github.com/user-attachments/assets/f3281089-88a4-43f5-a893-393e1d2bb5d6)
Pre-train results and denoised signal:
![image](https://github.com/user-attachments/assets/1f0de12e-526e-49bb-8716-8aa158e7ecab)
![image](https://github.com/user-attachments/assets/fce52b3a-a04b-4a36-8338-5cab64f9b9bd)

Final model combined the two by a simple concatenation. Experimented on the order of concatenation and dereverberation before denoising was having better results.
Hyperparameter tuning was done multiple time till reaching satisfying results both in MSE and by the enhanced signal hear test.
Noisy and reverbed signal:
![image](https://github.com/user-attachments/assets/f5d2d4b0-6e9e-4178-a4ff-12d88f4244ef)
Training results and cleaned signal:
![image](https://github.com/user-attachments/assets/068eec6f-8670-4196-980c-93e705cd56d4)
![image](https://github.com/user-attachments/assets/43506b36-831f-44b5-845f-afc227215298)

Additionaly used these metrics to evaluate the models results throughout the process:
* STOI: Short Term Objective Intelligibility 
* CD: Cepstral Distance 
* LLR: Log-Likelihood Ratio 
* fwSNRseg: Frequency Weighted Segmental SNR 
* SRMR: Speech to Reverberation Modulation Energy Ratio 
