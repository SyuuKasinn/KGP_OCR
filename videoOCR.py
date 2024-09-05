import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(15, 10))

# Define colors
colors = {
    'master_db': '#FF9999',
    'slave_db': '#FFCC99',
    'backend': '#99CCFF',
    'frontend': '#FFCCFF',
    'reporting_server': '#FFFF99',
    'search': '#CCFFCC',
    'kubernetes': '#CCCCFF',
}

# Draw rectangles
rects = {
    'master_db': mpatches.Rectangle((0.1, 0.7), 0.1, 0.1, edgecolor='black', facecolor=colors['master_db'],
                                    label='Master DB (PostgreSQL)'),
    'slave_db': mpatches.Rectangle((0.3, 0.7), 0.1, 0.1, edgecolor='black', facecolor=colors['slave_db'],
                                   label='Slave DB (PostgreSQL)'),
    'backend1': mpatches.Rectangle((0.1, 0.5), 0.1, 0.1, edgecolor='black', facecolor=colors['backend'],
                                   label='Backend Server 1'),
    'backend2': mpatches.Rectangle((0.3, 0.5), 0.1, 0.1, edgecolor='black', facecolor=colors['backend'],
                                   label='Backend Server 2'),
    'frontend': mpatches.Rectangle((0.1, 0.3), 0.3, 0.1, edgecolor='black', facecolor=colors['frontend'],
                                   label='Frontend (React)'),
    'reporting_server': mpatches.Rectangle((0.6, 0.7), 0.1, 0.1, edgecolor='black',
                                           facecolor=colors['reporting_server'], label='Reporting Server'),
    'search': mpatches.Rectangle((0.6, 0.5), 0.1, 0.1, edgecolor='black', facecolor=colors['search'],
                                 label='Search (Redis)'),
    'kubernetes': mpatches.Rectangle((0.55, 0.2), 0.2, 0.15, edgecolor='black', facecolor=colors['kubernetes'],
                                     label='Kubernetes'),
}

for rect in rects.values():
    ax.add_patch(rect)

# Draw arrows
arrows = [
    ((0.2, 0.75), (0.3, 0.75)),  # Master to Slave
    ((0.15, 0.7), (0.15, 0.6)),  # Master to Backend 1
    ((0.35, 0.7), (0.35, 0.6)),  # Slave to Backend 2
    ((0.2, 0.5), (0.2, 0.4)),  # Backend 1 to Frontend
    ((0.35, 0.5), (0.35, 0.4)),  # Backend 2 to Frontend
    ((0.65, 0.7), (0.35, 0.7)),  # Reporting Server to Slave
    ((0.65, 0.5), (0.35, 0.5)),  # Search to Backend 2
    ((0.65, 0.55), (0.65, 0.35)),  # Search to Kubernetes
    ((0.65, 0.75), (0.65, 0.35)),  # Reporting to Kubernetes
]

for start, end in arrows:
    print(start, end)
    ax.add_line(mlines.Line2D(*zip(start, end), color='black', linewidth=1, marker='>', markersize=5))

# Add texts
texts = {
    'master_db': (0.15, 0.75, 'Master DB\n(PostgreSQL)'),
    'slave_db': (0.35, 0.75, 'Slave DB\n(PostgreSQL)'),
    'backend1': (0.15, 0.55, 'Backend Server 1'),
    'backend2': (0.35, 0.55, 'Backend Server 2'),
    'frontend': (0.25, 0.35, 'Frontend\n(React)'),
    'reporting_server': (0.65, 0.75, 'Reporting Server'),
    'search': (0.65, 0.55, 'Search\n(Redis)'),
    'kubernetes': (0.65, 0.25, 'Kubernetes'),
}

for key, (x, y, text) in texts.items():
    ax.text(x, y, text, ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Add title
plt.title('System Architecture Diagram')

# Add legend
handles, labels = [], []
for rect in rects.values():
    handles.append(rect)
    labels.append(rect.get_label())
ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.75, 0.5))

# Remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig('architecture_diagram.png')
# Show plot
plt.show()
