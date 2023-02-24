# Classification of areas prone to forest fires

*Authors : Aimi Okabayashi, Thomas Boyer, David Dahan, Pierre-Aurélien Stahl, Pierre Personnat, Martial Gil*

This challenge was done as a project for the Master 2 Data Science (2022/2023), DATACAMP course

## Introduction

On August 8, with 187,114 fires, the record number of daily fires ever recorded worldwide was set. No region of the world has been spared.

Forest fires in the world have doubled in 20 years. Global warming as well as drought are hitting particularly hard in the United States where these disasters are growing at an exponential rate with in 2022, about 49,700 fires that occurred in the country, with a total of about 3 million hectares burned. By comparison, more than 785,000 hectares went up in smoke in Europe in 2022 and that is nearly 50 times the area burned in France in 2022 after a devastating summer of fires, although the USA represents in surface "only" about 18 times France.

Forest fires now ravage about 3 million hectares more each year than in 2001, an area equivalent to that of Belgium. As a result, the loss of forest cover due to fires is increasing by about 4% per year.

Today a UN publication calls on governments to adopt a new "Fire Ready Formula", with two-thirds of spending devoted to planning, prevention, preparedness and recovery, and the remaining third to response. Today, direct responses to wildfires typically receive more than half of related spending, while planning receives less than one percent, according to the UN's environmental program.

With this in mind, our study will focus on building a model that predicts the probability that a geographic location in the US will be a fire outbreak in order to simulate potential fire paths and assist in the prevention and control of these fires. In this regard, we have a dataset that contains about 20 features for each point of interest.

This project aims at constructing a machine learning model capable at **classifying point of areas prone to forest fires**. 

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [dedicated notebook](FINAL_RAMP_chalange.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
