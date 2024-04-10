# On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

### [Paper](dummy) | [Data](dummy) | [Poster](dummy)

[Rajat Modi](https://www.linkedin.com/in/rajat-modi-54377877?originalSubdomain=in)\*,
[Vibhav Vineet](https://scholar.google.com/citations?user=E_UlAVQAAAAJ&hl=en)\*,
[Yogesh Singh Rawat](https://scholar.google.com.sg/citations?user=D_JvEcwAAAAJ&hl=en),


This is the official implementation and dataset release for our **NeurIPS 2023 paper** "On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes".


## GLOM: Hinton's Islands of agreement. 

> "A Static Image is (A) Rather Boring Video- Dr. Geoff Hinton, Forward Forward Algorithm: Some Preliminary Investigations "


**Working Principle:** Let a boring image be repeated T=8 along the temporal axis. Then, it is pumped through a **VIDEO- transformer** and output values of lower attention layers are visualized via simple t-sNE clustering. We can see the islands. No fancy tricks. The net has been trained bottom-up for recognition only. **Islands have been observed now in transformers**. Official OpenReview Discussions can be found [here](https://openreview.net/forum?id=0cltUI2Sto&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FTrack%2FDatasets_and_Benchmarks%2FAuthors%23your-submissions)).



![Hinton's Islands of agreement](assets/island_hinton.gif)

