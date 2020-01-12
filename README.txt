Download the fastText model at (https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz)

DataGeneration contains the scripts for cleaning the various misspelling corpora. It converts them into a suitable format for the training.
- datasetFilter.py contains functions to clean the collated file, removing duplicates and samples which labels are not in the fastText vocabulary.
- EditDistanceAnalysis.py computes the edit distance distribution of the dataset.
- PrepASpell.py removes all samples in the dataset which mistakes cannot be identified by ASpell
- PrepGrammarly.py organise the data into folders which makes manual evaluation via Grammarly more convenient.

The Dataset folder
- misspelling source contains the original misspelling corpora
- collated.txt is the compiled dataset from the above corpora.
- contain the fasttextvocab folder

fasttextvocab folder contains all the data used for training and testing.
- folders 0,1,2,3,4 are the training folds used for cross-validation
- contextemb contains the bert embedding of the context
- training.txt contains the full training set, training_index.txt are indices of the training samples in the original fasttextvocab.txt
- test.txt contains the full test set, test_index.txt are indices of the test samples in the original fasttextvocab.txt
- test_aspell.txt, test_aspell_index.txt contain the test set after filtering by ASpell (done by DataGeneration/PrepASpell.py)
- test_AG.txt, test_AG_index.txt contain the test set after filtering by ASpell and Grammarly (filtering by grammarly is performed manually). This is used in the final evaluation
- results_full_mistakeemb.txt contains the suggestions generated using only the mistake embedding for the full test set
- results_full.txt contains the suggestions generated using the contextual spell corrector for the full test set
- results_AG.txt contains the suggestions generated using the contextual spell corrector for the AG test set


To replicate the training process:
1. dataSplit.py - used to split the dataset (fasttextvocab.txt) into training folds and test set
2. contextEmbedding.py - used to contain the 30522-dimension embedding for each sample in the dataset
3. PCA.py - generate the PCA module using for dimensionality reductions (using the training folds)
4. PCAEncode.py
    - perform dimensionality reduction on the fasttextvocab using the PCA module (trained on the fold) to produce reduced context embedding
    - use the fastText model to generate mistake and label embedding
5. train.py
    - Train the FFNN
    - Produce training and validation checkpoints
6. visualise.py
    - Plot the validation loss and accuracy vs epoch

To replicate testing:
1. testMistakeEmb.py - get performance of mistake embedding in isolation
2. testASpell.py - get performance of ASpell (note: require download of ASpell, script makes a call to aspell.bin)
3. test.py - get performance of proposed model (you will need the checkpoint 1500 under fasttextvocab/0/ckpt)
4. testEnsemble.py - get performance of the ensemble model (require ASpell)