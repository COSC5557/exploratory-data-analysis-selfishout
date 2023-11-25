# Exploratory Data Analysis

For this exercise, you will have a look at the raw data before doing any machine
learning. This is to familiarize yourself with the data, see if there are any
potential problems or obstacles for applying machine learning, and if and what
preprocessing may be helpful.

Your deliverable will be a report, written in a style that it
would be suitable for inclusion in an academic paper as the "Experimental
Setup" section or similar. If unsure, check an academic paper of your choice,
for example [this one](https://www.eecs.uwyo.edu/~larsko/papers/pulatov_opening_2022-1.pdf). The
level of detail should be higher than in a typical academic paper though. Your
report should be at most five pages, including references and figures but
excluding appendices. It should have the following structure:
- Introduction: What problem are you solving, how are you going to solve it.
- Results of the Analysis: Describe the data you're using, e.g. how many
  features and observations, what are you predicting, any missing values, etc.
  Include plots as appropriate.
- Code: Add the code you've used as a separate file.

There is no required format for the report. You could, for example, use an
iPython notebook. Your report should contain a lot of plots or tables to
illustrate the data. Describe any issues you found with the data.

## Data

We will have a look at the [Wine Quality
dataset](https://archive-beta.ics.uci.edu/dataset/186/wine+quality). Choose the
one that corresponds to your preference in wine. You may also use a dataset of
your choice, for example one that's relevant to your research.

Choose a small number of preprocessing methods (e.g. normalizing feature values)
and explore their application to the data. This could, for example, include
showing the distribution of feature values before and after preprocessing.

## Submission

Add your report and code to this repository. Bonus points if you can set up a
Github action to automatically run the code and generate the report!
