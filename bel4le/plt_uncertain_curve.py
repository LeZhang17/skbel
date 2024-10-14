import numpy as np
import matplotlib.pyplot as plt

# Load the data, skipping the header row
data = np.loadtxt('UncertainCurve.txt', skiprows=1)
data = data[:20, :]

index = 11
x = data[:, 4*(index-1)]      # x-axis data for the second set
y_mdn = data[:, 4*(index-1)+1]   # y_mdn data for the second set
y_cca = data[:, 4*(index-1)+2]   # y_cca data for the second set
y_prior = data[:, 4*(index-1)+3]  # y_prior data for the second set

# Plotting the data
plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y_cca, lw=1, label='CCA')
ax.plot(x, y_mdn, lw=1, label='MDN')
ax.plot(x, y_prior, lw=1, label='Prior')

# Setting labels and title
ax.set_xlabel('Confidence interval width')
ax.set_ylabel('Accuracy')

# Adding legend
ax.legend(loc='lower right')

# Enabling grid with custom settings
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout to prevent clipping
fig.tight_layout()

# Display the plot
plt.show()
