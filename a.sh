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
    python Node_Classfication_All.py --dprate 0.1 --dropout 0.2 --s 2 --K $K 
done