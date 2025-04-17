import torch
import torch.nn as nn
import pdb


class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0)):
        super(BasicConv3d, self).__init__()
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        )

    def forward(self, x):
        return self.conv(x)


def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x


class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        Block1 = []
        for i in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats))
        self.Block1 = nn.Sequential(*Block1)

        Block2 = []
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats))
        self.Block2 = nn.Sequential(*Block2)

        Block3 = []
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats))
        self.Block3 = nn.Sequential(*Block3)

        self.reduceF = BasicConv3d(wn, n_feats * 3, n_feats, kernel_size=1, stride=1)
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))

        conv1 = []
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        conv1.append(self.relu)
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        self.conv1 = nn.Sequential(*conv1)

        conv2 = []
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        conv2.append(self.relu)
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        self.conv2 = nn.Sequential(*conv2)

        conv3 = []
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        conv3.append(self.relu)
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))))
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, x):

        res = x
        x1 = self.Block1(x) + x
        x2 = self.Block2(x1) + x1
        x3 = self.Block3(x2) + x2

        x1, depth = _to_4d_tensor(x1, depth_stride=1)
        x1 = self.conv1(x1)
        x1 = _to_5d_tensor(x1, depth)

        x2, depth = _to_4d_tensor(x2, depth_stride=1)
        x2 = self.conv2(x2)
        x2 = _to_5d_tensor(x2, depth)

        x3, depth = _to_4d_tensor(x3, depth_stride=1)
        x3 = self.conv3(x3)
        x3 = _to_5d_tensor(x3, depth)

        x = torch.cat([self.gamma[0] * x1, self.gamma[1] * x2, self.gamma[2] * x3], 1)
        x = self.reduceF(x)
        x = self.relu(x)
        x = x + res

        x = self.Conv(x)
        return x


class MCNet(nn.Module):
    def __init__(self, scale=4, n_colors=128, n_feats=64, img='p'):
        super(MCNet, self).__init__()

        scale = scale
        n_colors = n_colors
        n_feats = n_feats
        n_conv = 1
        kernel_size = 3
        if img == 'p':
            band_mean = (0.09454724, 0.086288414, 0.080491179, 0.081265974, 0.083250979, 0.083979273, 0.084401618,
                         0.084112953, 0.084335804, 0.085097652, 0.085949058, 0.085468514, 0.085448931, 0.086030409,
                         0.086565592, 0.086657922, 0.087757197, 0.089064735, 0.090098055, 0.091314822, 0.092722444,
                         0.094531097, 0.095902051, 0.097363372, 0.099763666, 0.102678743, 0.104297737, 0.105214944,
                         0.105987221, 0.107551981, 0.109589808, 0.111053912, 0.112208435, 0.113786524, 0.115286386,
                         0.116597098, 0.117721343, 0.118840672, 0.119996467, 0.121134497, 0.121712647, 0.121875336,
                         0.122342858, 0.123222681, 0.124456521, 0.125500855, 0.125867974, 0.126013637, 0.126611137,
                         0.126550015, 0.127189605, 0.128149484, 0.128728437, 0.129342566, 0.129195024, 0.12883682,
                         0.128921682, 0.129116034, 0.129323738, 0.129274051, 0.129703033, 0.130468337, 0.131260194,
                         0.132202862, 0.133215684, 0.1345517, 0.137234566, 0.14090824, 0.145955343, 0.15103158,
                         0.156166931, 0.161580875, 0.166495836, 0.171337173, 0.176934928, 0.183337506, 0.189431231,
                         0.195471649, 0.200131918, 0.203971005, 0.207843346, 0.210788806, 0.211282535, 0.207036062,
                         0.205901359, 0.208755096, 0.21120492, 0.211688497, 0.21178592, 0.212415473, 0.213652537,
                         0.213217436, 0.212173354, 0.212466611, 0.212955782, 0.212439402, 0.211557078, 0.210355273,
                         0.20889418, 0.207323389, 0.204750736, 0.205432669)  #pavia

        else:
            band_mean = (0.005636499, 0.011562499, 0.01371138, 0.01647429, 0.017315993, 0.01922528, 0.01903994,
                         0.017904332, 0.017148501, 0.015892257, 0.01632191, 0.017315589, 0.018326869, 0.020198172,
                         0.019220744, 0.02073225, 0.020867041, 0.021160157, 0.022751569, 0.02431974, 0.024767126,
                         0.023937324, 0.024172657, 0.024089174, 0.02321178, 0.022968269, 0.022940352, 0.023443758,
                         0.024108071, 0.02622732, 0.027432103, 0.028451109, 0.030902818, 0.033267694, 0.035486394,
                         0.036517764, 0.03766751, 0.038922957, 0.040882701, 0.039744998, 0.039489859, 0.038675945,
                         0.038495487, 0.037199797, 0.03686058, 0.036199072, 0.036038524, 0.036198834, 0.036173574,
                         0.036251105, 0.035427776, 0.03530161, 0.034819624, 0.034751357, 0.034372505, 0.034136728,
                         0.033246776, 0.032656767, 0.031360668, 0.030273104, 0.029566403, 0.029593407, 0.030840691,
                         0.034100146, 0.042045962, 0.050919103, 0.060419419, 0.070224451, 0.080026666, 0.089812118,
                         0.099424797, 0.109055551, 0.118448257, 0.127387002, 0.135622065, 0.141036147, 0.144267517,
                         0.146839978, 0.148877917, 0.15051654, 0.151926809, 0.15120314, 0.152115409, 0.151761574,
                         0.151257328, 0.150677531, 0.150135159, 0.149733005, 0.148973658, 0.148322667, 0.148391809,
                         0.147963858, 0.147658236, 0.147469746, 0.149086846, 0.150134676, 0.145801667, 0.149169211,
                         0.149534894, 0.150000729, 0.149700322, 0.14932687, 0.149022387, 0.148032716, 0.147370742,
                         0.146342245, 0.146267613, 0.145974907, 0.14527817, 0.1447305, 0.14397888, 0.142890928,
                         0.134669083, 0.131493849, 0.126765871, 0.119772555, 0.116940664, 0.113350767, 0.117972406,
                         0.114139571, 0.11239062, 0.109074323, 0.105151379, 0.100168569, 0.094764662, 0.089503534,
                         0.084495308, 0.081443052)  # chikusei

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, n_colors, 1, 1])

        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size // 2))

        self.SSRM1 = Block(wn, n_feats, n_conv)
        self.SSRM2 = Block(wn, n_feats, n_conv)
        self.SSRM3 = Block(wn, n_feats, n_conv)
        self.SSRM4 = Block(wn, n_feats, n_conv)

        tail = []
        tail.append(
            wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3, 2 + scale, 2 + scale), stride=(1, scale, scale),
                                  padding=(1, 1, 1))))
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size // 2)))
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        x = x - self.band_mean.cuda()
        x = x.unsqueeze(1)
        T = self.head(x)

        x = self.SSRM1(T)
        x = torch.add(x, T)

        x = self.SSRM2(x)
        x = torch.add(x, T)

        x = self.SSRM3(x)
        x = torch.add(x, T)

        x = self.SSRM4(x)
        x = torch.add(x, T)

        x = self.tail(x)
        x = x.squeeze(1)
        x = x + self.band_mean.cuda()
        return x
