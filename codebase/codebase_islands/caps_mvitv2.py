#capsules containing no dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from pytorch_i3d import InceptionI3d
from slowfast.utils.parser import load_config, parse_args
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.models import build_model
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.logging as logging
from einops import reduce, rearrange, repeat

from pathlib import Path
import shutil

logger = logging.get_logger(__name__)


##########################   IGNORE THIS ####################################################
########## rmodi: this is just fluff, i apologize, cant remove it since layer name is in weights. 
class sentenceNet(nn.Module):

    def __init__(self):
        super(sentenceNet, self).__init__()

        self.conv1 = nn.Conv1d(300, 300, kernel_size=2, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=15, stride=1)

        self.conv2 = nn.Conv1d(300, 300, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=14, stride=1)


        self.conv3 = nn.Conv1d(300, 300, kernel_size=4, padding=0)
        self.pool3 = nn.MaxPool1d(kernel_size=13, stride=1)


        self.pool4 = nn.MaxPool1d(kernel_size=3)



        self.dense1 = nn.Linear(300, 136)
        self.dense1.weight.data.normal_(0.0, 1.0)



        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        h1 = self.relu(self.conv1(input))
        h1 = self.pool1(h1)

        h2 = self.relu(self.conv2(input))
        h2 = self.pool2(h2)

        h3 = self.relu(self.conv3(input))
        h3 = self.pool3(h3)

        h1 = torch.cat([h1, h2, h3],dim=2)
        h1 = self.pool4(h1)

        h1 = h1.view(-1,300)
        h1 = self.relu(self.dense1(h1))

       
        return h1
########## rmodi: this is just fluff, i apologize, cant remove it since layer name is in weights. 
# will take care in future, 
##########################   IGNORE ENDS ####################################################


################ USEFUL CODE FROM HERE #######################################################

class primarySentenceCaps(nn.Module):
    def __init__(self):
        super(primarySentenceCaps, self).__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        p = x[:,0:128]
        p = p.view(-1,16,8)
        
        a = x[:,128:136]
        a = a.view(-1,1,8)
        a = self.sigmoid(a)
       
        out = torch.cat([p, a], dim=1)
       
        out = out.permute(0, 2, 1)
        
        return out
########## rmodi: this is just fluff, i apologize, cant remove it since layer name is in weights. 
# will take care in future, 
#PrimaryCaps(832, 8, 9, P, stride=1)
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    # def __init__(self, A=32, B=64, K=9, P=4, stride=1):
    def __init__(self,A, B, K, P, stride):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.pose.weight.data.normal_(0.0, 0.1)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.a.weight.data.normal_(0.0, 0.1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out

#ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B, C, K, P, stride, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        # self._lambda = 1e-03
        self._lambda = 1e-6
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsul/home/bruce/projects/capsulese type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.randn(C,self.psize))
        self.beta_a = nn.Parameter(torch.randn(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K[0]*K[1]*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        cost_h = cost_h.sum(dim=2)


        cost_h_mean = torch.mean(cost_h,dim=1,keepdim=True)

        cost_h_stdv = torch.sqrt(torch.sum(cost_h - cost_h_mean,dim=1,keepdim=True)**2 / C + eps)
        

        a_out = self.sigmoid(self._lambda*(self.beta_a - (cost_h_mean -cost_h)/(cost_h_stdv + eps)))

        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(eps + a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for h_idx in range(0, h - K + 1, stride)] \
                for k_idx in range(0, K)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def add_pathes2(self,x, B, K=(3, 3), psize=4, stride=(1, 1)):
        b, h, w, c = x.shape
        assert c == B * (psize + 1)

        oh = int((h - K[0] + 1) / stride[0])
        ow = int((w - K[1] + 1) / stride[1])

        idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - K[0] + 1, stride[0])] for k_idx in range(0, K[0])]
        idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - K[1] + 1, stride[1])] for k_idx in range(0, K[1])]

        x = x[:, idxs_h, :, :]
        x = x[:, :, :, idxs_w, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            # x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)
            x, oh, ow = self.add_pathes2(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()

            p_in=p_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, self.psize)
            a_in = a_in.view(b * oh * ow, self.K[0] * self.K[1] * self.B, 1)
            # p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            # a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            # print('conv cap activations',a_out[0].sum().item(),a_out[0].size())
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K[0] and 1 == self.K[1]
            assert 1 == self.stride[0] and 1 == self.stride[1]
            # assert 1 == self.K
            # assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


class CapsNet(nn.Module):


    def __init__(self,cfg=None, P=4, pretrained_load=False):
        super(CapsNet, self).__init__()
        self.P = P
        self.cluster_init = False
        
        self.conv1 = build_model(cfg)
        print("loading weights...")
        if cfg.TRAIN.CHECKPOINT_FILE_PATH!='':
            checkpoint_epoch = cu.load_checkpoint(
                cfg.TRAIN.CHECKPOINT_FILE_PATH,
                self.conv1,
                cfg.NUM_GPUS > 1,
                optimizer=None,
                scaler=None,
                inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
                convert_from_caffe2=False,
                epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
                image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
            )
        print("loaded weights...")
        self.conv1 = self.conv1.to(torch.device("cuda"))
        
        self.primary_caps = PrimaryCaps(384, 32, 9, P, stride=1)
        #self.conv_caps = ConvCaps(16, 8, (1, 1), P, stride=(1, 1), iters=3)
        self.conv_caps = ConvCaps(32, 21, (1, 1), P, stride=(1, 1), iters=3)

        #self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1 = nn.ConvTranspose2d(336, 64, kernel_size=9, stride=1, padding=0)
        self.upsample1.weight.data.normal_(0.0, 0.02)

        self.upsample2 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample2.weight.data.normal_(0.0, 0.02)


        #self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=1,output_padding=(0,1,1))
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding=1)
        self.upsample3.weight.data.normal_(0.0, 0.02)

        self.upsample4 = nn.ConvTranspose3d(128, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=1, output_padding = (1,1,1))
        self.upsample4.weight.data.normal_(0.0, 0.02)
        
        self.dropout3d = nn.Dropout3d(0.5)

        self.smooth = nn.ConvTranspose3d(128, 1, kernel_size=3,padding = 1)
        self.smooth.weight.data.normal_(0.0, 0.02)


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.sentenceNet = sentenceNet()
        self.sentenceCaps = primarySentenceCaps()

        self.conv28 = nn.Conv2d(384, 64, kernel_size=(3, 3), padding=(1, 1))

        self.conv56 = nn.Conv3d(192, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv112 = nn.Conv3d(96, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    
        self.conv_cluster = nn.Conv2d(408, 408, kernel_size = (3, 3), padding=(1, 1))        


    def load_pretrained_weights(self):
        saved_weights = torch.load('./savedweights/weights_referit')
        self.load_state_dict(saved_weights, strict=False)
        print('loaded referit pretrained weights for whole network')

    def load_previous_weights(self, weightfile):
        saved_weights = torch.load(weightfile)
        print("krishna jhmdb capsules....")
        self.load_state_dict(saved_weights, strict=True)
        print('loaded weights from previous run: ', weightfile)
        print(saved_weights.keys())
        # exit(1)
    def catcaps(self, wordcaps, imgcaps):
        num_wordcaps = wordcaps.size()[1]
        num_word_poses = int(wordcaps.size()[2] - 1)
        h = imgcaps.size()[1]
        w = imgcaps.size()[2]
        img_data = imgcaps.size()[3]
        num_imgcaps = int(img_data / (self.P * self.P))
        wordcaps = torch.unsqueeze(wordcaps, 1)
        wordcaps = torch.unsqueeze(wordcaps, 1)
        word_caps = wordcaps.repeat(1, h, w, 1, 1)

        word_poses = word_caps[:, :, :, :, :num_word_poses]
        word_poses = word_poses.contiguous().view(-1, h, w, num_wordcaps * num_word_poses)

        word_acts = word_caps[:, :, :, :, num_word_poses]
        word_acts = word_acts.view(-1, h, w, num_wordcaps)

        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, word_poses, img_acts, word_acts), dim=-1)
        return combined_caps
    
    def caps_reorder(self, imgcaps):
        h = imgcaps.size()[1]
        w = imgcaps.size()[2]
        img_data = imgcaps.size()[3]
        num_imgcaps = int(img_data / (self.P * self.P))
        
        pose_range = num_imgcaps * self.P * self.P
        img_poses = imgcaps[:, :, :, :pose_range]
        img_acts = imgcaps[:, :, :, pose_range:pose_range + num_imgcaps]

        combined_caps = torch.cat((img_poses, img_acts), dim=-1)
        return combined_caps
        
    def forward_island(self, img):
        # gave some love to mvitv2 in videomodelbuilder.py in slowfast
        _, cross112,cross56, cross28, cross14 = self.conv1(img)
        cross112 = cross112.detach().cpu()
        return cross112
    
    