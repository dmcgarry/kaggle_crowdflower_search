./train_doc2vec.py --name word --punc --digits --html
./train_doc2vec.py --name lancaster --stem lancaster --punc --digits --html
./train_doc2vec.py --name porter --stem porter --punc --digits --html
./train_doc2vec.py --name snowball --stem snowball --punc --digits --html
./train_word2vec.py --name word --punc --digits --html
./train_word2vec.py --name lancaster --stem lancaster --punc --digits --html
./train_word2vec.py --name porter --stem porter --punc --digits --html
./train_word2vec.py --name snowball --stem snowball --punc --digits --html
./train_dtm.py --name pca_svm_porter --grid pca_svm.txt --stem porter --punc --digits --html --label
./train_dtm.py --name pca_reg_svm --grid dtm_pca_reg_svm.txt --reg --stem porter --html --label --punc --seed 90
./train_dtm.py --name stacked_reg --grid stacked_dtm.txt --reg --stem snowball --html --label --punc --digits --seed 33
./train_dtm.py --name lda_reg_svm --grid lda_reg_svm.txt --reg --stemming porter --html --label --punc --digits --seed 77
./ensemble.py --cut 0.6 --var 0.5 --lower 2.1 --upper 3.4


