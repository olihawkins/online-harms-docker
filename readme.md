# online-harms-docker

This is an adaptation of the [OnlineHarms-Metatool](https://github.com/amittos/OnlineHarms-Metatool), which functions as a standalone application that runs in a Docker container. 

This version of the metatool supports the Davidson and Wulcyzn models. Minor changes to the code have been introduced to support CSV rather than TSV as the input and output format.

## Build

To build the Docker container from source use:

```
docker build -t online-harms .
```

## Use

The tool will classify a CSV of texts using two models trained on a range of datasets. To use the tool, place a CSV containing the texts to classify in a directory for the `dataset`. The CSV should contain a single column of texts, stored in a column named `text`. When the tool is run, the results of the classifiers will be written to a `results` directory. You will need to tell the Docker container which directories to use for `dataset` and `results` when you run it.

To run the Docker container use:

```
docker run \
    -v ${PWD}/dataset:/app/dataset \
    -v ${PWD}/results:/app/results \
    online-harms
```

