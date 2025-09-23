#!/usr/bin/env python3
"""
Test script to verify datashader multi-color clustering functionality.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test datashader categorical aggregation
def test_datashader_clustering():
    try:
        import datashader as ds
        import colorcet as cc

        # Create test data with multiple clusters
        np.random.seed(42)
        n_points = 5000

        # Cluster 1
        x1 = np.random.normal(0, 0.5, n_points//3)
        y1 = np.random.normal(0, 0.5, n_points//3)
        cluster1 = np.zeros(len(x1), dtype=int)

        # Cluster 2
        x2 = np.random.normal(2, 0.3, n_points//3)
        y2 = np.random.normal(2, 0.3, n_points//3)
        cluster2 = np.ones(len(x2), dtype=int)

        # Noise points (cluster -1)
        x3 = np.random.uniform(-2, 4, n_points//3)
        y3 = np.random.uniform(-2, 4, n_points//3)
        cluster3 = np.full(len(x3), -1, dtype=int)

        # Combine all data
        x_all = np.concatenate([x1, x2, x3])
        y_all = np.concatenate([y1, y2, y3])
        clusters_all = np.concatenate([cluster1, cluster2, cluster3])

        # Create DataFrame with categorical cluster data
        df = pd.DataFrame({
            'x': x_all,
            'y': y_all,
            'cluster': pd.Categorical(clusters_all)
        })

        print(f"Created test data: {len(df)} points with clusters {df['cluster'].unique()}")

        # Create datashader canvas
        cvs = ds.Canvas(plot_width=400, plot_height=400)

        # Categorical aggregation
        agg = cvs.points(df, 'x', 'y', agg=ds.by('cluster', ds.count()))

        # Define color mapping
        color_key = {
            -1: 'black',    # Noise
            0: 'blue',      # Cluster 0
            1: 'red'        # Cluster 1
        }

        # Create shaded image
        img = ds.tf.shade(agg, color_key=color_key, how='eq_hist')
        img_pil = img.to_pil()

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Regular scatter plot
        for cluster_id in [-1, 0, 1]:
            mask = clusters_all == cluster_id
            if cluster_id == -1:
                label, color = 'Noise', 'black'
            else:
                label, color = f'Cluster {cluster_id}', color_key[cluster_id]
            ax1.scatter(x_all[mask], y_all[mask], c=color, alpha=0.6, s=2, label=label)
        ax1.set_title('Regular Scatter Plot')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Datashader plot
        ax2.imshow(img_pil, extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
                  aspect='auto', origin='lower')
        ax2.set_title('Datashader Multi-Color')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        plt.tight_layout()
        plt.savefig('/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Code/Python/pyBayerSMLM/datashader_clustering_test.png', dpi=150)
        plt.show()

        print("✅ Datashader categorical aggregation test passed!")
        print(f"Saved test plot to: datashader_clustering_test.png")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install: pip install datashader colorcet")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_datashader_clustering()