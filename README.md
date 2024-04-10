# On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

### [Paper](dummy) | [Data](dummy) | [Poster](dummy)

[Rajat Modi](https://www.linkedin.com/in/rajat-modi-54377877?originalSubdomain=in)\*,
[Vibhav Vineet](https://scholar.google.com/citations?user=E_UlAVQAAAAJ&hl=en)\*,
[Yogesh Singh Rawat](https://scholar.google.com.sg/citations?user=D_JvEcwAAAAJ&hl=en),


This is the official implementation and dataset release for our **NeurIPS 2023 paper** "On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes".


## GLOM: Hinton's Islands of agreement. 

> "A Static Image is (A) Rather Boring Video- Dr. Geoff Hinton, Forward Forward Algorithm: Some Preliminary Investigations "


**Working Principle:** Let a boring image be repeated T=8 along the temporal axis. Then, it is pumped through a **VIDEO- transformer** and output values of lower attention layers are visualized via simple t-sNE clustering. We can see the islands. No fancy tricks. The net has been trained bottom-up for recognition only. **Islands have been observed now in transformers**. Official OpenReview Discussions can be found [here](https://openreview.net/forum?id=0cltUI2Sto&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FTrack%2FDatasets_and_Benchmarks%2FAuthors%23your-submissions)).

<div align="center">
<img src="assets/island_hinton.gif" alt="Hinton's Islands of agreement" width="600" height="300">
</div>

<div align="center">
<img src="assets/island_dataset.gif" alt="Island Datasets" width="600" height="300">
</div>

## Limitations of capsules.

>" The fundamental weakness of capsules is that they use a mixture to
model the set of possible parts. This forces a hard decision about whether a
car headlight and an eye are really different parts. If they are modeled by the
same capsule, the capsule cannot predict the identity of the whole. If they are
modeled by different capsules the similarity in their relationship to their whole
cannot be captured. If we want to make neural networks that understand images in the same way as people do, we need to figure out how neural networks can represent part-whole hierarchies. This is difficult because a real neural network cannot dynamically
allocate a group of neurons to represent a node in a parse tree
. The inability of neural nets to dynamically allocate neurons was the motivation for a
series of models that used “capsules”- Dr. Geoff Hinton, GLOM "

We now confirm that capsules undergo collapse if too many objects are present in the scene. This is problem with all other models including transformers: their memory increases with number of objects in the scene.

<div align="center">
<img src="assets/collapse.png" alt="Hinton's Islands of agreement" width="600" height="300">
</div>


## Dataset Release.

What happens when number of objects in the scene scale up? To answer this, we release all the datasets used in this paper. 

<div align="center">
<img src="assets/benchmark_statistics.png" alt="Hinton's Islands of agreement" width="600" height="300">
</div>

