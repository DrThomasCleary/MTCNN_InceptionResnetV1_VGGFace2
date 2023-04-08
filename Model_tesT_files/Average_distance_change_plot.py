import matplotlib.pyplot as plt
import numpy as np

# Data
square_block_test = ['5%', '10%', '15%', '20%']
matched_faces = [0.8374373216997401, 0.8710947963059318, 1.0058749722072537, 1.217308804818562]
mismatched_faces = [1.3254281334646518, 1.276977905898843, 1.2975618596738498, 1.32998401560131743]

# Line plot
fig, ax = plt.subplots()

# Define plot title
plot_title = "MTCNN/InceptionresnetV1(VGGFace2) \n Change in Average Distance with increase in Square Size for Occlusion testing"

ax.plot(square_block_test, matched_faces, marker='o', linestyle='-', label='Matched Faces')
ax.plot(square_block_test, mismatched_faces, marker='s', linestyle='-', label='Mismatched Faces')

ax.set_xlabel('Square Size Relative To Image')
ax.set_ylabel('Average Distance')
ax.set_title(plot_title)
ax.legend()

plt.show()
