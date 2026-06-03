"""
Elliptic dataset preprocessor for EvolveGCN.
Usage:
    python preprocess_elliptic.py --input_dir /path/to/raw/kaggle/files --output_dir /path/to/data/elliptic

Input files expected in input_dir:
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv

Output files written to output_dir:
    elliptic_txs_features.csv          (modified in place)
    elliptic_txs_classes.csv           (modified in place)
    elliptic_txs_orig2contiguos.csv    (new)
    elliptic_txs_nodetime.csv          (new)
    elliptic_txs_edgelist_timed.csv    (new)
"""

import argparse
import csv
import os

def preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    features_in  = os.path.join(input_dir,  'elliptic_txs_features.csv')
    classes_in   = os.path.join(input_dir,  'elliptic_txs_classes.csv')
    edges_in     = os.path.join(input_dir,  'elliptic_txs_edgelist.csv')

    features_out     = os.path.join(output_dir, 'elliptic_txs_features.csv')
    classes_out      = os.path.join(output_dir, 'elliptic_txs_classes.csv')
    orig2cont_out    = os.path.join(output_dir, 'elliptic_txs_orig2contiguos.csv')
    nodetime_out     = os.path.join(output_dir, 'elliptic_txs_nodetime.csv')
    edges_out        = os.path.join(output_dir, 'elliptic_txs_edgelist_timed.csv')

    # ------------------------------------------------------------------
    # Step 1: Process features + build orig->contiguous id map
    # ------------------------------------------------------------------
    print("Step 1: Processing features and building id map...")
    orig2cont = {}   # original string id -> new integer id
    node_times = {}  # new integer id -> timestep (0-based)

    new_features_rows = []
    orig2cont_rows    = [['originalId', 'contiguosId']]
    nodetime_rows     = [['txId', 'timestep']]

    with open(features_in, 'r') as f:
        reader = csv.reader(f)
        for new_id, row in enumerate(reader):
            orig_id  = row[0].strip()
            timestep = int(row[1].strip()) - 1   # shift to 0-based

            # build maps
            orig2cont[orig_id] = new_id
            node_times[new_id] = timestep

            # modified features row: replace orig_id with float new_id,
            # replace timestep with float timestep, keep rest unchanged
            new_row = [str(float(new_id)), str(float(timestep))] + [v.strip() for v in row[2:]]
            new_features_rows.append(new_row)

            orig2cont_rows.append([orig_id, str(new_id)])
            nodetime_rows.append([str(new_id), str(timestep)])

    with open(features_out, 'w', newline='') as f:
        csv.writer(f).writerows(new_features_rows)

    with open(orig2cont_out, 'w', newline='') as f:
        csv.writer(f).writerows(orig2cont_rows)

    with open(nodetime_out, 'w', newline='') as f:
        csv.writer(f).writerows(nodetime_rows)

    print(f"  {len(new_features_rows)} nodes processed.")

    # ------------------------------------------------------------------
    # Step 2: Process classes
    # ------------------------------------------------------------------
    print("Step 2: Processing classes...")
    label_map = {'unknown': '-1.0', '1': '1.0', '2': '0'}
    classes_rows = [['txId', 'class']]

    with open(classes_in, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            orig_id   = row[0].strip()
            label_str = row[1].strip()
            new_id    = orig2cont[orig_id]
            new_label = label_map[label_str]
            classes_rows.append([str(float(new_id)), new_label])

    with open(classes_out, 'w', newline='') as f:
        csv.writer(f).writerows(classes_rows)

    print(f"  {len(classes_rows) - 1} class labels processed.")

    # ------------------------------------------------------------------
    # Step 3: nodetime already written in Step 1
    # ------------------------------------------------------------------
    print("Step 3: nodetime file already written in Step 1.")

    # ------------------------------------------------------------------
    # Step 4: Process edgelist
    # ------------------------------------------------------------------
    print("Step 4: Processing edgelist...")
    edges_rows = [['txId1', 'txId2', 'timestep']]
    skipped = 0

    with open(edges_in, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            orig1 = row[0].strip()
            orig2 = row[1].strip()

            if orig1 not in orig2cont or orig2 not in orig2cont:
                skipped += 1
                continue

            new1 = orig2cont[orig1]
            new2 = orig2cont[orig2]
            ts   = node_times[new1]

            # sanity check: both nodes should share the same timestep
            if node_times[new1] != node_times[new2]:
                print(f"  WARNING: edge ({orig1},{orig2}) spans timesteps "
                      f"{node_times[new1]} and {node_times[new2]} — skipping.")
                skipped += 1
                continue

            edges_rows.append([str(new1), str(new2), str(float(ts))])

    with open(edges_out, 'w', newline='') as f:
        csv.writer(f).writerows(edges_rows)

    print(f"  {len(edges_rows) - 1} edges written, {skipped} skipped.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nDone. Output files:")
    for fname in [
        'elliptic_txs_features.csv',
        'elliptic_txs_classes.csv',
        'elliptic_txs_orig2contiguos.csv',
        'elliptic_txs_nodetime.csv',
        'elliptic_txs_edgelist_timed.csv',
    ]:
        path = os.path.join(output_dir, fname)
        size = os.path.getsize(path)
        print(f"  {fname}  ({size:,} bytes)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True,
                        help='Directory containing raw Kaggle CSV files')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to write preprocessed files (will be created)')
    args = parser.parse_args()
    preprocess(args.input_dir, args.output_dir)
