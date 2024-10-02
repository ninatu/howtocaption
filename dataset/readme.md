# HowToCaption Dataset

[**[arxiv]**](https://arxiv.org/abs/2301.02009)

**The HowToCaption dataset** comprises
1.2M long-term instructional videos from [the HowTo100M dataset](https://www.di.ens.fr/willow/research/howto100m/), 
where ASR subtitles have been transformed into proper captions
via our HowToCaption method using [the Vicuna-13B LLM](https://lmsys.org/blog/2023-03-30-vicuna/) ([v0](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md)). 
The captions are automatically generated 
and their high-quality alignment to the video are further 
ensured through subsequent alignment and filtering post-processing, 
all achieved without any human involvement. As a result, the HowToCaption dataset contains 25M aligned video-text pairs.

Get the dataset:
* **HowToCaption dataset** (video_id+captions+timestamps): [Link](https://drive.google.com/file/d/1GU6G29RcVO8Og9D5CsJDS24kRs3bpJ2a/view?usp=drive_link) (~1.5GB) 
* **Unfiltered version** with 
the corresponding similarity scores of caption to video clip (video_id+captions+timestamps+scores): [Link](https://drive.google.com/file/d/1Do_anJj-FB8lGINKbUgbj7vd8AQaByzY/view?usp=drive_link) (~4.6GB)

Additionally, we provide the HowToCaption-grounded dataset, featuring captions obtained via [the MiniGPT4 model](https://minigpt-4.github.io/):
* **HowToCaption-grounded dataset** (video_id+captions+timestamps): [Link](https://drive.google.com/file/d/1zBXyCHgO8zrytd1m3ohq3eF3nFEwAMki/view?usp=drive_link) (~1.5GB)
* **Unfiltered version** with 
the corresponding similarity scores of caption to video clip (video_id+captions+timestamps+scores): [Link](https://drive.google.com/file/d/1uNqxfEgviOt-Fmr9qMb0D3OhZ3LlqeIv/view?usp=drive_link) (~4.5GB) 

### How To Use 

#### How to use filtered HowToCaption or HowToCaption-grounded datasets:  
* Each file is a dictionary with video-ids as keys 
* For each video we provide ‘start’, ‘end’, and ‘text’ lists of the same lengths  
* ’start’ and ‘end’ correspond to starts and ends seconds of the clips in the video

  
To note: 
- ‘text’ is list of lists of strings as to the same position in the video can correspond several captions
- Starting seconds in ‘start’ list are not ordered; however, ‘end’ seconds always correspond  to ’start’ positions ordering


**Example**:   

```
<<< HowToCaption[‘---39MFGZ-k’]   

{
'start': [12, 19, 29, 25, 55, 81, 82], 
'end': [20, 27, 37, 33, 63, 89, 90], 
'text': [
[‘Show how to unload a 12-gauge shotgun’], 
[‘Loading a 12-gauge shotgun’], 
[‘Demonstrating how to unload a 12-gauge shotgun', 'A better way to unload a gun’], 
[‘Putting another round into the gun', 'The danger of loading a gun the usual way’], 
[‘Loading the gun safely', 'Short stroke to load the gun', 'Loading the gun today’], 
[‘Lifting up the bar to extract rounds’], 
[‘Going forward and lifting up the bar to extract rounds'] 
}
```

#### How to use unfiltered HowToCaption or HowToCaption-grounded datasets:  

The difference to standard HowToCaption dataset is that ‘text’ is list of lists of tuples of (string, score).

**Example**:
```
<<< HowToCaption[‘---39MFGZ-k’]

{
'start': [12, 19, 25, 29, 55, 54, 65, 81, 82, 105, 103], 
'end': [20, 27, 33, 37, 63, 62, 73, 89, 90, 113, 111], 
'text': [
[('Show how to unload a 12-gauge shotgun', 0.5699871778488159)], 
[('Loading a 12-gauge shotgun', 0.5876383185386658)], 
[('Unloading and removing a round from the chamber', 0.31276029348373413), ('Putting another round into the gun', 0.4805337190628052), ('The danger of loading a gun the usual way', 0.4611629843711853)],
[('Demonstrating how to unload a 12-gauge shotgun', 0.617999255657196), ('A better way to unload a gun', 0.5126216411590576)], 
[('Loading the gun safely', 0.539146363735199), ('Short stroke to load the gun', 0.5076732635498047), ('Loading the gun today', 0.4759426712989807)], 
[('Being nervous on camera', 0.3465729355812073), ('Nervousness on camera', 0.27738460898399353)], 
[('Extracting rounds by lifting up the bar', 0.41076189279556274)], 
[('Lifting up the bar to extract rounds', 0.4220432639122009)], 
[('Going forward and lifting up the bar to extract rounds', 0.42620745301246643)], 
[('A person is speaking and pointing out that there are no ramps present', 0.30187565088272095)], 
[('The speaker mentions that they can be found online', 0.30197498202323914), ('The speaker concludes the video by saying "WWE" and ending the video', 0.36031144857406616)]]
}
```

### Acknowledgement
* [BLIP](https://github.com/salesforce/BLIP) is the model for text-video encoder and score function
* [Vicuna](https://github.com/lm-sys/FastChat/tree/main) is open source instructional LLM to generate HowToCaption dataset
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) is open-source LLM with image conditioning  to generate HowToCaption-grounded dataset


If you're using HowToCaption or HowToCaption-grounded dataset in your research or applications, please cite using this BibTeX:

```
@article{shvetsova2023howtocaption,
  title={HowToCaption: Prompting LLMs to Transform Video Annotations at Scale},
  author={Shvetsova, Nina and Kukleva, Anna and Hong, Xudong and Rupprecht, Christian and Schiele, Bernt and Kuehne, Hilde},
  journal={ECCV},
  year={2024}
}
```


### Licence: 

HowToCaption and HowToCaption-grounded are based on Vicuna and MiniGpt-4 that are fine-tuned LLaMA and should be used under [LLaMA's model license](https://github.com/facebookresearch/llama/blob/main/LICENSE).


