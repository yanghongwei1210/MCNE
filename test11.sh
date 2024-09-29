for K in $(seq 10 -1 1)
do
    # 运行Python脚本，并传递K参数 
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.1 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.1 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.1 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.2 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.2 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.2 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.2 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.3 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.3 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.3 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.4 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.4 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.4 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.4 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.5 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.3 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.5 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.5 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.5 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.6 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.6 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.6 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.6 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.7 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.7 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.7 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.7 --dropout 0.8 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.1 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.1 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.1 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.1 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.2 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.2 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.2 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.2 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.2 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.3 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.3 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.3 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.3 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.3 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.4 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.4 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.4 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.4 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.4 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.5 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.5 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.5 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.5 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.5 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.6 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.6 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.6 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.6 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.6 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.7 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.7 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.7 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.7 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.7 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.8 --s 0 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.8 --s 1 --K $K &&
    python Node_Classfication_All.py --dprate 0.8 --dropout 0.8 --s 2 --K $K &&
    python adapt_multi.py --t 0 --dprate 0.8 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 1 --dprate 0.8 --dropout 0.8 --K $K &&
    python adapt_multi.py --t 2 --dprate 0.8 --dropout 0.8 --K $K     
done