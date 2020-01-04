import typing
import numpy
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from environ import Player

num_filters = 64
num_blocks = 5

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other

  def __gt__(self, other):
    return self.index > other

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [i for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    if len(self.history) % 2 == 0:
      return Player.white
    else:
      return Player.black

# Nets
class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h

body_out_shape = (num_filters, 10, 9)

class Prediction(nn.Module):
    ''' Policy and value prediction from inner abstract state '''

    def __init__(self, action_shape):
        super().__init__()
        self.board_size = 90
        self.action_size = action_shape

        self.conv_policy = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, self.action_size)
        )

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, rp):
        batch_size = rp.size()[0]
        pol = self.conv_policy(rp)
        pol = self.policy(pol.view(batch_size, -1))

        h_v = F.relu(self.conv_v(rp))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # range of value is -1 ~ 1
        return F.softmax(pol, dim=-1), torch.tanh(h_v)

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        return int(numpy.prod(o.size()))


class Dynamics(nn.Module):
    '''Abstruct state transition'''

    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.layer0(h)
        for block in self.blocks:
            h = block(h)
        return h


class Network(nn.Module):

    def __init__(self, action_space_size: int, device):
        super().__init__()
        self.steps = 0
        self.action_space_size = action_space_size
        self.device = device
        input_shape = (14, 10, 9)
        rp_shape = (num_filters, *input_shape[1:])
        self.representation = Representation(input_shape).to(device)
        self.prediction = Prediction(action_space_size).to(device)
        self.dynamics = Dynamics(rp_shape, (num_filters, 10, 9)).to(device)
        self.eval()

    def predict_initial_inference(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (14, 10, 9) or x.shape[1:] == (14, 10, 9)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 14, 10, 9)

        x = torch.Tensor(x).to(self.device)
        h = self.representation(x)
        policy, value = self.prediction(h)

        if orig_x.ndim == 3:
            return h[0], policy[0], value[0]
        else:
            return h, policy, value

    def predict_recurrent_inference(self, x, a):

        if x.ndim == 3:
            x = x.reshape(1, num_filters, 10, 9)

        a = numpy.full((1, num_filters, 10, 9), a)

        g = self.dynamics(x, torch.Tensor(a).to(self.device))
        policy, value = self.prediction(g)

        return g[0], policy[0], value[0]

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        iat = image.astype(numpy.float32)
        h, p, v = self.predict_initial_inference(iat)
        return NetworkOutput(v, 0, p, h)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        g, p, v = self.predict_recurrent_inference(hidden_state, action)
        return NetworkOutput(v, 0, p, g)

