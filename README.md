# Overview

This project and associated repository served two purposes.  The first and most basic was to develop strong knowledge of Python tools for data analytics.  The only coding experience I had in data analytics previously was in R.  In addition, this project served as my introduction to Github.

The second purpose of this project was to analyze the market for data jobs, specifically those in the United States.  Analysis includes trends in skill demand, salary, job titles, and more.  The data used for this analysis is sourced from Luke Barrouse's Python Course

## The Questions

I will be answering the following questionsL

1.  What are the skills most in-demand for the top three most popular data roles in the US?

2.  How are in-demand skills trending for Data Analysts over a year?

3.  How well do jobs and skills pay for Data Analysts?

4.  What are the optimal skills for Data Analysts to learn?


## Tools I used

- Python:  
    - The first "real" programming language I've learned, for those who don't like to count R. Libraries used included Pandas for data analysis, matplotlib and Seaborn for visualizations, and scipy/numpy for some more robust analysis.

- Jupyter Notebooks:  Tool used for running python scripts in small pieces.

- Visual Studio Code:  The first proper IDE I've used, for executing python scripts.

- Git and Github:  For version control and easy publishing of work.



## Data cleaning and preparation

```python

#Importing Libraries

import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
import ast
import seaborn as sns


#Loading Data

dataset = load_dataset('lukebarousse/data_jobs')
df = dataset['train'].to_pandas()

#Data Cleanup

df['job_posted_date'] = pd.to_datetime(df['job_posted_date'])

df['job_skills'] = df['job_skills'].apply(lambda skill_list:  ast.literal_eval(skill_list) if isinstance(skill_list, str) else skill_list)

```
These lines are in almost every notebook of this project, and were used to clean and prep the data for proper analysis.  Other cleaning methods, like dropNA, were used as needed.

# Analysis

Each notebook in the 'project' folder answers a specific question related to the data job market.  

## 1.  What are the most demanded skills for the top three most popular data roles?

To answer this question I first identified which roles were the most popular, then identified the top five most demanded skills in those roles.  This step gave me an idea of which skills I should learn after Python, depending on which roles I am targeting.

For more detail on the code used, view my notebook here:  [2_skill_demand.ipynb](3_project\2_skill_demand.ipynb)

### Visualize Data

```python

fig, ax = plt.subplots(len(job_titles), 1)

sns.set_theme(style='ticks')

for i, job_title in enumerate(job_titles):
    df_plot = df_skills_pct[df_skills_pct['job_title_short']==job_title].head(5)
    sns.barplot(data=df_plot, x='skill_percent', y='job_skills', ax=ax[i], hue='skill_count', palette='flare')
    ax[i].set_title(job_title)
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    ax[i].legend().set_visible(False)
    ax[i].set_xlim(0,80)
    ax[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x)}%'))
    for  n, v in enumerate(df_plot['skill_percent']):
        ax[i].text(v+1, n, f'{v:.0f}%', va='center')
    
    if i !=len(job_titles)-1:
        ax[i].set_xticks([])

fig.suptitle("Likelihood of Skills Requested in US Job Postings", fontsize=15)
fig.tight_layout(h_pad=0.5)


plt.show()

```
### Results

![Visualization of Top Skills demanded for common data roles](3_project\images\skill_demand_all_data_roles.png)

### Insights

1.  SQL is in high demand across many roles, appearing in no less than 51% of postings for the big three.

2.  Learning Python qualifies me for data scientist and data engineer positions.  I'd been avoiding these due to so many of them demanding Python, but I'm now confident I could do the job.

3.  My R skills were not a waste, it's called for in many data scientist jobs.  Some of them even ask applicants to know Python and R.

4.  From this graph, we can see that prospective data engineers should learn at least one cloud technology.