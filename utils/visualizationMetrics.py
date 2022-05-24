"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""
"""
Revised by Qilong Pan: 2022 4-7
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import io
import torch
import os

   
def visualization (ori_data, generated_data, analysis, save_name, epoch, args):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    """  
    # Analysis sample size (for faster computation)
    anal_sample_no = min([args.eval_num, len(ori_data), len(generated_data), 500])

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    if type(generated_data) is not np.ndarray:
        if type(generated_data) is torch.Tensor:
            generated_data = np.asarray(generated_data.to('cpu'))
    elif type(generated_data) is np.ndarray:
        generated_data = np.asarray(generated_data)
    else:
        print('The generated is not either torch.tensor, nor np.ndarray. Please check your data type for the output of generator.')
        raise TypeError

    no, channles, seq_len = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 0), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],0), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(ori_data[i,:,:],0), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                        np.reshape(np.mean(generated_data[i,:,:],0), [1,seq_len])))
    
    # Visualization parameter        
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components = 2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        ## fixed x,y axis
        x_min = np.min(pca_results[:, 0]) * args.swell_ratio
        x_max = np.max(pca_results[:, 0]) * args.swell_ratio
        y_min = np.min(pca_results[:, 1]) * args.swell_ratio
        y_max = np.max(pca_results[:, 1]) * args.swell_ratio
        f, ax = plt.subplots(1)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

        ax.legend(loc = 1)
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
#         plt.show()

    elif analysis == 'tsne':

        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
        x_min = np.min(tsne_results[:anal_sample_no, 0]) * args.swell_ratio
        x_max = np.max(tsne_results[:anal_sample_no, 0]) * args.swell_ratio
        y_min = np.min(tsne_results[:anal_sample_no, 1]) * args.swell_ratio
        y_max = np.max(tsne_results[:anal_sample_no, 1]) * args.swell_ratio
        # Plotting
        f, ax = plt.subplots(1)
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                    c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
        plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                    c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
        ax.legend(loc=1)
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
#         plt.show()

    plt.savefig(os.path.join(args.path_helper['log_path_img_pca'],f'{save_name}_epoch_{epoch+1}.png'), format="png")
#    plt.show()