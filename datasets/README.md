# Datasets

Place dataset files under this folder with the structure:

```text
datasets/
  <dataset_name>/
    <data_name>.mat
    <adjacency_name>.mat
```

For example:

```text
datasets/
  noaa_coastwatch_cellular/
    data_oriented_mov.mat
    adjacencies_oriented.mat
  noaa_coastwatch_edge/
    data.mat
    adjacencies.mat
```

`examples/utils/data_utils.py` uses this folder as the primary lookup path.
