import torch
import torch.nn as nn
import torch.nn.functional as F
import util

"""
文章同时挖掘交通图的节点和边的特征，借用了2019年《hypergraph Nerual Network》文章中超图提取边特征的思想，引进hypergraph帮助提取边特征.
"""

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,wv->ncwl', (x, A))
        return x.contiguous()


class d_nconv(nn.Module):
    def __init__(self):
        super(d_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,nvw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class linear_(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear_, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=2, padding=(0, 0), stride=(1, 1),
                                   bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)           # torch.Size([1, 40, 207, 64])
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2                     # # torch.Size([1, 40, 207, 64])
        h = torch.cat(out, dim=1)           # torch.Size([1, 120, 207, 64])
        h = self.mlp(h)                     # torch.Size([1, 1, 207, 64])
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# dgcn结gcn一样，名为Dynamic只是因为使用了HGCN的动态信息
class dgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order * 3 + 1) * c_in           # c_in = 280
        self.mlp = linear_(c_in, c_out)         # Conv2d(280, 20, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
        self.dropout = dropout   # dropout = 0.3
        self.order = order                      

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.d_nconv(x, a)                 # torch.Size([64, 40, 207, 11])
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)            # torch.Size([64, 40, 207, 11])
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)                   # torch.Size([64, 280, 207, 11])
        h = self.mlp(h)                             # torch.Size([64, 280, 207, 9])
        h = F.dropout(h, self.dropout, training=self.training)      # torch.Size([64, 280, 207, 9])
        return h


# HGCN是用的Hypergraph NN文章中的计算框架，计算得到信息，用于Dynamic Graph Convolution中DGCN
class hgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(hgcn, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class hgcn_edge_At(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=1):
        super(hgcn_edge_At, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.nconv(x, a)           # torch.Size([64, 40, 1722, 1])
            out.append(x1)                  # torch.Size([64, 40, 1722, 1])
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)      # torch.Size([64, 40, 1722, 1])
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)           # torch.Size([64, 80, 1722, 1])
        h = self.mlp(h)                     # torch.Size([64, 1, 1722, 1]) 
        h = F.dropout(h, self.dropout, training=self.training)  # torch.Size([64, 1, 1722, 1]) 
        return h


# dhgcn结构同hgcn一样，名为Dynamic只是因为使用了GCN的动态信息
class dhgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dhgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear_(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.d_nconv(x, a)             # torch.Size([64, 40, 1083, 11])
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)        # torch.Size([64, 40, 1083, 11])
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)               # torch.Size([64, 120, 1083, 11])
        h = self.mlp(h)                         # torch.Size([64, 20, 1083, 9])
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class spatial_attention(nn.Module):
    def __init__(self, in_channels, num_of_timesteps, num_of_edge, num_of_vertices):
        super(spatial_attention, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()                     # (13)
        self.W2 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()                     # (13)
        self.W3 = nn.Parameter(torch.randn(in_channels,int(in_channels/2)).cuda(), requires_grad=True).cuda()       # (40,20)
        self.W4 = nn.Parameter(torch.randn(in_channels,int(in_channels/2)).cuda(), requires_grad=True).cuda()       # (40,20)
        self.out_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=(1, 1))                                                               # Conv2d(40, 40, kernel_size=(1, 2), stride=(1, 1))

    def forward(self, x, idx, idy):
        # x: torch.Size([64, 207, 40, 13])
        # idx: torch.Size([1722])
        # idy: torch.Size([1722])
        lhs = torch.matmul(torch.matmul(x, self.W1),self.W3)        # torch.Size([64, 207, 20])
        rhs = torch.matmul(torch.matmul(x, self.W2),self.W4)        # torch.Size([64, 207, 20]) 
        sum = torch.cat([lhs[:,idx,:], rhs[:,idy,:]], dim=2)        # torch.Size([64, 1722, 40])
        sum = torch.unsqueeze(sum, dim=3).transpose(1, 2)           # torch.Size([64, 40, 1722, 1]) 
        S = self.out_conv(sum)                                      # torch.Size([64, 40, 1722, 1]) 
        S = torch.squeeze(S).transpose(1, 2)                        # torch.Size([64, 1722, 40])
        return S


class ddstgcn(nn.Module):
    def __init__(self, batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl,num_nodes,
                 dropout=0.3, supports=None, in_dim=2, out_dim=12, residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640, kernel_size=2, blocks=3, layers=1):
        super(ddstgcn, self).__init__()
        self.batch_size = batch_size    # batch_size = 64
        self.H_a = H_a                  # (1083, 207)
        self.H_b = H_b                  # (1083, 207)
        self.G0 = G0                    # (1083, 207)
        self.G1 = G1                    # (207, 1083)
        self.H_T_new = H_T_new          # (207, 1083)
        self.lwjl = lwjl                # torch.Size([64, 1, 1083, 1])
        self.indices = indices          # (2,1722)
        self.G0_all = G0_all            # (1722,207)
        self.G1_all = G1_all            # (207, 1722)


        self.edge_node_vec1 = nn.Parameter(torch.rand(self.H_a.size(1), 10).cuda(), requires_grad=True).cuda()  # (207, 10)
        self.edge_node_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(0)).cuda(), requires_grad=True).cuda()  # (10, 1083)

        self.node_edge_vec1 = nn.Parameter(torch.rand(self.H_a.size(0), 10).cuda(), requires_grad=True).cuda()  # (1083, 10)
        self.node_edge_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(1)).cuda(), requires_grad=True).cuda()  # (10, 207)

        self.hgcn_w_vec_edge_At_forward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(),
                                                       requires_grad=True).cuda()                               # (207)
        self.hgcn_w_vec_edge_At_backward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(),
                                                        requires_grad=True).cuda()                              # (207)

        self.dropout = dropout          # dropout = 0.3
        self.blocks = blocks            # blocks = 3
        self.layers = layers            # layers = 1

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgconv = nn.ModuleList()
        self.filter_convs_h = nn.ModuleList()
        self.gate_convs_h = nn.ModuleList()
        self.SAt_forward = nn.ModuleList()
        self.SAt_backward = nn.ModuleList()
        self.hgconv_edge_At_forward = nn.ModuleList()
        self.hgconv_edge_At_backward = nn.ModuleList()
        self.gconv_dgcn_w = nn.ModuleList()
        self.dhgconv = nn.ModuleList()
        self.bn_g = nn.ModuleList()
        self.bn_hg = nn.ModuleList()


        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))                      # Conv2d(2, 40, kernel_size=(1, 1), stride=(1, 1))
        self.supports = supports            # len(supports) = 2, supports[0].shape = torch.Size([207, 207])
        self.num_nodes = num_nodes          # num_nodes = 207
        receptive_field = 1                 
        self.supports_len = 0
        self.supports_len += len(supports)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()  # (207,10)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()  # (10,207)
        self.supports_len += 1              # self.supports_len = 3

        for b in range(blocks):   # blocks = 3
            additional_scope = kernel_size   # kernel_size = 2
            new_dilation = 2                 
            for i in range(layers):  # 1
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))   # Conv2d(40, 40, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))     # Conv2d(40, 40, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))                                      # Conv2d(40, 320, kernel_size=(1, 1), stride=(1, 1))
                self.bn.append(nn.BatchNorm2d(residual_channels))                                          # BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.filter_convs_h.append(nn.Conv2d(in_channels=1+residual_channels*2,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))  # Conv2d(81, 40, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
                self.gate_convs_h.append(nn.Conv2d(in_channels=1+residual_channels*2,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))    # Conv2d(81, 40, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))

                self.SAt_forward.append(spatial_attention(residual_channels, int(13-receptive_field+1),
                                                                 self.indices.size(1), num_nodes))          # residual_channels = 40, int(13-receptive_field+1) = 13, self.indices.size(1) = 1722, num_nodes =207
                self.SAt_backward.append(spatial_attention(residual_channels, int(13-receptive_field+1),
                                                                  self.indices.size(1), num_nodes))
                
                receptive_field += (additional_scope * 2)   # receptive_field = 5

                self.dgconv.append(dgcn(dilation_channels, int(residual_channels / 2), dropout))            # dilation_channels = 40, int(residual_channels / 2) = 20, # Conv2d(280, 20, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
                self.hgconv_edge_At_forward.append(hgcn_edge_At(residual_channels, 1, dropout))             # (mlp): Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))
                self.hgconv_edge_At_backward.append(hgcn_edge_At(residual_channels, 1, dropout))            # (mlp): Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1))
                self.gconv_dgcn_w.append(
                    gcn((residual_channels), 1, dropout, support_len=2, order=1))                           # (mlp): Conv2d(120, 1, kernel_size=(1, 1), stride=(1, 1))
                self.dhgconv.append(dhgcn(dilation_channels, int(residual_channels / 2), dropout))          # (mlp): Conv2d(120, 20, kernel_size=(1, 2), stride=(1, 1), dilation=(2, 2))
                self.bn_g.append(nn.BatchNorm2d(int(residual_channels / 2)))                                # BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.bn_hg.append(nn.BatchNorm2d(int(residual_channels / 2)))                               # BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)                  # Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)                  # Conv2d(640, 12, kernel_size=(1, 1), stride=(1, 1))

        self.receptive_field = receptive_field                  # 13
        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)    # BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.new_supports_w = [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]    # torch.Size([64, 207, 207])
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]    # torch.Size([64, 207, 207]) 
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]    # torch.Size([64, 207, 207])


    def forward(self, input):
        in_len = input.size(3)      # input: torch.Size([64, 2, 207, 13]), in_len = 13
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.bn_start(x)        # torch.Size([64, 2, 207, 13])
        x = self.start_conv(x)      # torch.Size([64, 40, 207, 13])
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)      # (207,207)
        adp_new = adp.repeat([self.batch_size, 1, 1])                               # torch.Size([64, 207, 207])
        new_supports = self.supports + [adp]                                        # (207,207)
        edge_node_H = (self.H_T_new * (torch.mm(self.edge_node_vec1, self.edge_node_vec2)))   # (207, 1083)
        self.H_a_ = (self.H_a * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))         # (1083, 207)
        self.H_b_ = (self.H_b * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))         # # (1083, 207)
        G0G1_edge_At_forward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_forward))) @ self.G1_all        # (1722, 1722)
        G0G1_edge_At_backward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_backward))) @ self.G1_all       # (1722, 1722)
        self.new_supports_w[2] = adp_new.cuda()                                                 # torch.Size([64, 207, 207])
        forward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()       # torch.Size([64, 207, 207])
        backward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()      # torch.Size([64, 207, 207])

        for i in range(self.blocks * self.layers):    # 3*1
            # Dual Transformation -- Graph to Hypergraph Transformation （8）
            edge_feature = util.feature_node_to_edge(x, self.H_a_, self.H_b_, operation="concat")               # torch.Size([64, 80, 207, 13])
            edge_feature = torch.cat([edge_feature, self.lwjl.repeat(1, 1, 1, edge_feature.size(3))], dim=1)    # torch.Size([64, 81, 207, 13])

            # Dynamic Hypergraph Convolution -- Gate-TCN (15)
            filter_h = self.filter_convs_h[i](edge_feature)         # torch.Size([64, 40, 1083, 11])
            filter_h = torch.tanh(filter_h)                         # torch.Size([64, 40, 1083, 11])
            gate_h = self.gate_convs_h[i](edge_feature)             # torch.Size([64, 40, 1083, 11])
            gate_h = torch.sigmoid(gate_h)                          # torch.Size([64, 40, 1083, 11])
            x_h = filter_h * gate_h                                 # torch.Size([64, 40, 1083, 11])


            # Dynamic Interaction Module (14)
            # DIM中Extraction：筛选始发Node和终止Node，形成新的input_data
            batch_edge_forward = self.SAt_forward[i](x.transpose(1,2),self.indices[0],self.indices[1])          # torch.Size([64, 1722, 40])
            batch_edge_backward = self.SAt_backward[i](x.transpose(1,2),self.indices[0],self.indices[1])        # torch.Size([64, 1722, 40])
            batch_edge_forward = torch.unsqueeze(batch_edge_forward, dim=3)                                     # torch.Size([64, 1722, 40, 1])
            batch_edge_forward = batch_edge_forward.transpose(1,2)                                              # torch.Size([64, 40, 1722, 1])
            # HGCN
            batch_edge_forward = self.hgconv_edge_At_forward[i](batch_edge_forward, G0G1_edge_At_forward)       # torch.Size([64, 40, 1722, 1])
            batch_edge_forward = torch.squeeze(batch_edge_forward)                                              # torch.Size([64, 1, 1722, 1])
            forward_medium[:,self.indices[0],self.indices[1]] = torch.sigmoid((batch_edge_forward))
            self.new_supports_w[0] = forward_medium
            batch_edge_backward = torch.unsqueeze(batch_edge_backward, dim=3)                                   # torch.Size([64, 1722, 40, 1])
            batch_edge_backward = batch_edge_backward.transpose(1, 2)                                           # torch.Size([64, 40, 1722, 1])
             # HGCN
            batch_edge_backward = self.hgconv_edge_At_backward[i](batch_edge_backward, G0G1_edge_At_backward)   # torch.Size([64, 1, 1722, 1])
            batch_edge_backward = torch.squeeze(batch_edge_backward)                                            # torch.Size([64, 1722])
            backward_medium[:,self.indices[0],self.indices[1]] = torch.sigmoid((batch_edge_backward))           # 
            self.new_supports_w[1] = backward_medium.transpose(1,2)                                             # torch.Size([64, 207, 207])
            self.new_supports_w[0] = self.new_supports_w[0] * new_supports[0]                                   # torch.Size([64, 207, 207])
            self.new_supports_w[1] = self.new_supports_w[1] * new_supports[1]                                   # torch.Size([64, 207, 207])

            # Dynamic Graph Convolution -- Gate-TCN (15)
            residual = x                                    # torch.Size([64, 40, 207, 13])
            filter = self.filter_convs[i](residual)         # torch.Size([64, 40, 207, 11])
            filter = torch.tanh(filter)                     
            gate = self.gate_convs[i](residual)             # torch.Size([64, 40, 207, 11])
            gate = torch.sigmoid(gate)
            x = filter * gate                               # torch.Size([64, 40, 207, 11])
            # Dynamic Graph Convolution -- DGCN (16)
            x = self.dgconv[i](x, self.new_supports_w)      # torch.Size([64, 20, 207, 9])
            x = self.bn_g[i](x)                             # torch.Size([64, 20, 207, 9])

            # Dynamic Interaction Module -- GCN (11)
            dhgcn_w_input = residual                        # torch.Size([64, 40, 207, 13])
            dhgcn_w_input = dhgcn_w_input.transpose(1, 2)   # torch.Size([64, 207, 40, 13])
            dhgcn_w_input = torch.mean(dhgcn_w_input, 3)    # torch.Size([64, 207, 40])
            dhgcn_w_input = dhgcn_w_input.transpose(0, 2)   # torch.Size([40, 207, 64])
            dhgcn_w_input = torch.unsqueeze(dhgcn_w_input, dim=0)   # torch.Size([1, 40, 207, 64])
            dhgcn_w_input = self.gconv_dgcn_w[i](dhgcn_w_input, self.supports)  # torch.Size([1, 1, 207, 64])
            dhgcn_w_input = torch.squeeze(dhgcn_w_input)        # torch.Size([207, 64])
            dhgcn_w_input = dhgcn_w_input.transpose(0, 1)       # torch.Size([64, 207])
            
            # Dynamic Hypergraph Convolution -- DHGCN (19)
            # dhgcn_w_input
            dhgcn_w_input = self.G0 @ (torch.diag_embed(dhgcn_w_input)) @ self.G1       # torch.Size([64, 1083, 1083])
            x_h = self.dhgconv[i](x_h, dhgcn_w_input)                                   # torch.Size([64, 40, 1083, 11])
            x_h = self.bn_hg[i](x_h)                                                    # torch.Size([64, 20, 1083, 9])

            # Dual Transformation -- Hypergraph to Graph Transformation  (9)
            x = util.fusion_edge_node(x, x_h, edge_node_H)                              # torch.Size([64, 40, 207, 9])

            # Concatenation operation
            x = x + residual[:, :, :, -x.size(3):]                                      # torch.Size([64, 40, 207, 9])
            x = self.bn[i](x)

            # Skip Connection
            s = x                                           # torch.Size([64, 40, 207, 9])
            s = self.skip_convs[i](s)                       # torch.Size([64, 320, 207, 9])
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip                                 # torch.Size([64, 320, 207, 9])

        x = F.leaky_relu(skip)
        x = F.leaky_relu(self.end_conv_1(x))

        # Linear Transformation
        x = self.end_conv_2(x)                              # torch.Size([64, 640, 207, 1])

        return x


