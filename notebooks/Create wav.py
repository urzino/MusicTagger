
# coding: utf-8

# In[5]:


from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm
import os


# In[6]:


annotation = pd.read_csv('../data/MagnaTagATune/annotations_final.csv', sep='\t')


# In[ ]:


with tqdm(total=len(list(annotation['mp3_path']))) as pbar:
    for value in annotation['mp3_path']:
        input_song = "../data/MagnaTagATune/mp3/" + value
        try:
            output_song = "../data/MagnaTagATune/rawwav_2/" + value[:-3] + 'wav'
            output_dir = '../data/MagnaTagATune/rawwav_2/' + value[:2]
            if not os.path.exists(os.path.dirname(output_dir)):
                os.makedirs(os.path.dirname(output_dir))
            if not os.path.exists(output_song):
                song = AudioSegment.from_mp3(input_song)
                song.set_frame_rate(22050)
                song.export(output_song, format="wav")
            pbar.update(1)
        except:
            print(value)

