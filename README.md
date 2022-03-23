# Predicting the Potential Acceptance of a COVID-19 Vaccine on Social Data

Authors : Oumaima Marbouh, Yasmine Guemouria, Daniel Quintão de Moraes, Mehdy Bennani, Júlia Togashi, Yousra Leouafi

This challenge was done as a project for the Master 2 Data Science (2021/2022), DATACAMP course

## Introduction

The coronavirus pandemic, is an ongoing pandemic of a respiratory disease caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). On March 11, 2020, the World Health Organization (WHO) classified the outbreak as a Pandemic. In phase III development studies, several vaccines against COVID-19 have demonstrated efficacy of up to 95% in preventing symptomatic infections of the disease. As of March 2021, 12 vaccines have been authorized by at least one national regulatory authority for public use.

As we know, **acceptance of the vaccine** is not the same for everyone and varies according to various social factors. Better understanding the characteristics that lead someone to be unwilling to get the vaccine is a public health necessity. Moreover, the ability to individually predict an individual's likelihood to be vaccinated can be applied in various scenarios, such as for government-directed advertising or document control at borders between countries.

This project aims at constructing a machine learning model capable at predicting the answer to two questions: 
1.  **If a COVID-19 vaccine is proven safe and effective and is available to me, I will take it**
2.  **I would accept a vaccine if it were recommended by my employer and was approved safe and effective by the government**. 

Each question has 5 possible answers, ranging from "completely disagree" to "completely agree", as further described below.

Thus, we have a multitarget **and** multiclass classification problem.

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