import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.conv_network = nn.Sequential(
            # in channels -> the "depth" of the data entering the layer
            # we have 4 in channels because greyscale is 1 channel, but we are stacking 4 frames

            # out channels -> the number of filters to apply (the number of unique features to scan for)

            # kernel_size -> the size of the scanner (eg 8x8)

            # stride -> how far the scanner should jump after each scan

            # the specific numbers here have come from https://www.nature.com/articles/nature14236

            # the first conv layer takes in 4 channels (4 different frames)
            # it then returns 32 different 20x20 grids, with each grid detecting different features (20x20 comes from
            # the 8x8 scanner going across the screen and snapshotting, then jumping 4 pixels and repeat)
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),

            # now we take all 32 of those 20x20 grids in, and send our 4x4 scanner across, detecting 64 unique features.
            # this produces 64 9x9 grids, with each 9x9 grid detecting a different, larger feature
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),

            # now we take all 64 of those 9x9 grids in, send our 3x3 scanner across, again detecting 64, unique, again
            # larger features, producing 64 7x7 grids
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),

            # now we have 64 7x7 grids. each of these grids is a highly specialised detector for a specific thing
            # on the screen. Finally, we convert these 64 7x7 grids back to 64x7x7 (3136) decimal numbers,
            # ready to be inputted into our Linear layers with nn.Flatten()
            nn.Flatten()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),

            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),

            nn.Linear(512, output_size)
        )

    def forward(self, x):
        features = self.conv_network(x)

        v = self.value_stream(features)
        a = self.advantage_stream(features)

        a = a - torch.mean(a, dim=1, keepdim=True)

        q = v + a
        return q
