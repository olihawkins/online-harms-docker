# online-harms-docker

This is an adaptation of the [OnlineHarms-Metatool](https://github.com/amittos/OnlineHarms-Metatool), which functions as a standalone application that runs in a Docker container. 

This version of the metatool supports the Davidson and Wulcyzn models. Minor changes to the code have been introduced to support CSV rather than TSV as the input and output format. The metatool also now expects and returns a column of document ids for each text to be classified.

## Build

To build the Docker container from source use:

```
docker build -t online-harms .
```

## Use

The tool will classify a CSV of texts using two models trained on a range of datasets. To use the tool, place a CSV containing the texts to classify in a directory for the `dataset`. 

The CSV should contain two columns:

- A column called `id`, which contains an id for each document.
- A column called `text`, which contains the text of each document.

When the tool is run, the results of the classifiers will be written to a `results` directory. You will need to tell the Docker container which directories to use for `dataset` and `results` when you run it.

To run the Docker container use:

```
docker run \
    -v ${PWD}/dataset:/app/dataset \
    -v ${PWD}/results:/app/results \
    online-harms
```

By default the tool predicts binary classes for each of the input texts, with 0 indicating unharmful text and 1 indicating harmful text. The ensemble prediction represents the mode of the predictions from each of the different classifiers. 

You can instead have the classifiers predict the probability that the text is harmful with the `-p` argument. In this case the ensemble prediction represents the mean of the predictions from each of the different classifiers.

```
docker run \
    -v ${PWD}/dataset:/app/dataset \
    -v ${PWD}/results:/app/results \
    online-harms -p
```