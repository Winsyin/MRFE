# MRFE
Multi-resolution sequence and structure feature extraction for binding site prediction
# Dependency 
pytorch 1.10  
sklearn 
python 3.7  
genism 3.8.3

# Content 
./dataset: All circRNA-RBP datasets, including sequences, predicted base pairings, and bpRNA annotations.

# Notice
circRNA word vector model trained by iCircRBP-DHN(https://academic.oup.com/bib/article/22/4/bbaa274/5943796?login=true) can be downloaded from https://github.com/houzl3416/iCircRBP-DHN.

# Usage
1.Unzip the RBP dataset under./dataset, then simply run the following command to train MRFE.

python train.py --protein RBP_name --modelType your_circRNA2Vec_model_path --num_levels number_of_forward_residual_SCI-Block_stacks

2.To predict the secondary structure of circRNAs, you should ensure that RNAfold is installed, which can be downloaded from https://www.tbi.univie.ac.at/RNA/, using the tutorial found at https://www.tbi.univie.ac.at/RNA/tutorial/#sec3_1.

