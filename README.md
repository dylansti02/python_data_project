# Overview

This project and associated repository served two purposes.  The first and most basic was to develop strong knowledge of Python tools for data analytics.  The only coding experience I had in data analytics previously was in R.  In addition, this project served as my introduction to Github.

The second purpose of this project was to analyze the market for data jobs, specifically those in the United States.  Analysis includes trends in skill demand, salary, job titles, and more.  The data used for this analysis is sourced from Luke Barrouse's Python Course

## The Questions

I will be answering the following questionsL

1.  What are the skills most in-demand for the top three most popular data roles in the US?

2.  How are in-demand skills trending for Data Analysts over a year?

3.  How well do jobs and skills pay for Data Analysts?

4.  What are the optimal skills for Data Analysts, Data Scientists, and Data Engineers to learn?


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

## 2.  How are in-demand skills trending for Data Analysts over a year?

To answer this question I first performed some pandas operations to get the percentage of job postings containing a certain skill by month.  I then plotted the top five skills in a line chart with Seaborn.

For more detail on the code used, view my notebook here:  [3_skills_trend.ipynb](3_project\3_skills_trend.ipynb)


### Visualize Data

```python
sns.lineplot(data=df_plot, dashes=False, palette='Set2')
sns.set_theme(style='ticks')

plt.title('Trending Top Skills for Data Analysts in the US')
plt.ylabel('Likelihood in Job Posting')
plt.xlabel('2023')
plt.legend().remove()

from matplotlib.ticker import PercentFormatter

ax=plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

for i in range(5):
    
    if i==3:
         plt.text(9.2, df_plot.iloc[-1, i]+2, df_plot.columns[i])   
    elif i==2:
         plt.text(8.2, df_plot.iloc[-1, i]-3.8, df_plot.columns[i])
    else:    
        plt.text(11.2, df_plot.iloc[-1, i], df_plot.columns[i])


sns.despine()


```

### Results

![A line graph showing trends in skill demand over 2023](3_project\images\trending_skills.png)

### Insights

1.  There aren't many strong trends here.  SQL seems to drop in the fall, but there isn't any obvious reason for that to happen.

2.  That said, we can see excel drop following the end of intern season, with some of those interns being given full time jobs.

3.  Python is consistently in demand throughout the year.


## 3.  How well do jobs and skills pay for Data Analysts?

#Visualize Data

```python

sns.boxplot(data=df_us_top6, x='salary_year_avg', y='job_title_short', order=job_order)
sns.set_theme(style='ticks')


plt.title('Salary Distribution in the United States')
plt.xlabel('Yearly Salary')
plt.ylabel('')
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:  f'${int(x/1000)}k'))
plt.xlim(0, 700000)

plt.show()

```

```python

fig, ax = plt.subplots(2, 1)

sns.set_theme(style='ticks')



sns.barplot(data=df_da_top_pay, x='median', y=df_da_top_pay.index, ax=ax[0], hue='median', palette='dark:b_r')

ax[0].legend().remove()
ax[0].set_title('Top 10 Highest Paid Skills for Data Analysts')
ax[0].set_ylabel('')
ax[0].set_xlabel('')
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:  f'${int(x/1000)}k'))




sns.barplot(data=df_da_skills, x='median', y=df_da_skills.index, ax=ax[1], hue='median', palette='light:b')

ax[1].legend().remove()
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_title('Top 10 Most In-Demand Skills for Data Analysts')
ax[1].set_ylabel('')
ax[1].set_xlabel('Median Salary (USD)')
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:  f'${int(x/1000)}k'))
fig.tight_layout()
plt.show()

```

More detail on the code can be found here:  [4_salary_analysis.ipynb](3_project\4_salary_analysis.ipynb)

### Results

![Distribution of Salaries for top data roles in the US](3_project\images\salary_distribution.png)



### Insights

1.  Lower-experience data scientists and engineers are paid more than even senior data analysts.

2.  Skills in the top-paying skills graph seem to be more specialized, but it should be noted the sample size is very small.  It seems unlikely that an R user could double their potential salary by learning dplyr.

3.  Python is the highest-paying skill among the 10 most demanded skills.  

4.  The Microsoft Office big three are the lowest paying in the top 10, probably because they are so widely learned across disciplines.

## 4.  What are the optimal skills for Data Analysts, Data Scientists, and Data Engineers to learn?

### Analyze and Visualize Data

This question was answered using three essentially idential scripts, so I'll just show one of them:

```python

df_ds_us = df[(df['job_title_short']=='Data Scientist') & (df['job_country']=='United States')].copy()
df_ds_us=df_ds_us.dropna(subset=['salary_year_avg'])

df_ds_us_explode = df_ds_us.explode('job_skills')

df_ds_us_explode[['salary_year_avg', 'job_skills']].head(5)
df_ds_skills=df_ds_us_explode.groupby('job_skills')['salary_year_avg'].agg(['count', 'median']).sort_values(by='count', ascending=False)

df_ds_skills=df_ds_skills.rename(columns={'count':'skill_count', 'median':'median_salary'})

ds_job_count = len(df_ds_us)
df_ds_skills['skill_percent']=df_ds_skills['skill_count']/ds_job_count*100
skill_percent=8

df_ds_skills_high_demand = df_ds_skills[df_ds_skills['skill_percent']>skill_percent]

df_ds_skills_high_demand=df_ds_skills_high_demand.merge(df_technology, left_on='job_skills', right_on='skills')

from adjustText import adjust_text
from matplotlib.ticker import PercentFormatter

sns.scatterplot(
    data=df_ds_skills_high_demand, 
    x='skill_percent', 
    y='median_salary', 
    hue='technology'
    )

sns.despine()
sns.set_theme(style='ticks')

df_ds_skills_high_demand=df_ds_skills_high_demand.set_index('skills')

texts=[]


for i, txt in enumerate(df_ds_skills_high_demand.index):
    texts.append(plt.text(df_ds_skills_high_demand['skill_percent'].iloc[i], df_ds_skills_high_demand['median_salary'].iloc[i], txt ))
    

adjust_text(texts, arrowprops = dict(arrowstyle = "->", color = "gray", lw=1))



ax=plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos:  f'${int(y/1000)}k'))
ax.xaxis.set_major_formatter(PercentFormatter( decimals=0))

plt.xlabel('Percent of Data Scientist Jobs')
plt.ylabel('Median Yearly Salary')
plt.title('Most Optimal Skills for Data Scientists in the US')
plt.tight_layout()



plt.show()

```

More detail on the code can be found here:  [5_optimal_skills.ipynb](3_project\5_optimal_skills.ipynb)


I did alter the skill percent threshold for legibility of the result graphs, aiming for 10-15 skills per graph.  Speaking of which:

### Results

![Graph of Optimal Skills to learn for Data Analysts](3_project\images\da_skills.png)

![Graph of Optimal Skills to learn for Data Scientists](3_project\images\ds_skills.png)

![Graph of Optimal Skills to learn for Data Engineers](3_project\images\de_skills.png)

### Insights

1.  Upon seeing these graphs, I thought I'd investigate specialization of skills by checking for a negative correlation between skill percent and salary.  Results were not significant.

2.  Companies are looking for Python, but they are willing to pay much more for those who know machine learning libraries that can run in Python.

3.  Visualization tools are always in demand, but seem to be at less of a premium than programming tools.

4.  It seems like data scientists are data analysts who know machine learning, and are paid a lot more for it.  I'm sure that's an oversimplification.

5.  There is a market for those who know visualization tools without programming tools.  Most analyst jobs demand at least one.

6.  I'm very glad I decided to take the time to learn Python, it's the second most demanded skill overall.  This leads directly into my bonus question:

## 5.  Was it worth it?

Clearly, the qualitative answer to that question is yes.  There are 6 graphs on this page that'll tell you that.  Being a statistician, however, I want something stronger than a qualitative answer.  This led me to a t-test of the mean salary for jobs in the big three roles that require python versus those who do not.

### Visualize Data and Test Mean

Information on this code can by found at the bottom of this notebook:  [1_eda_intro.ipynb](3_project\1_eda_intro.ipynb)

Once again, we have three very similar scripts, so I'll just be showing one:

```python

from scipy import stats
import numpy as np
import statistics

df_de_us_nona = df_de_us.dropna(subset=['job_skills'])

df_de_us_explode = df_de_us_nona.explode('job_skills')


python_jobs = []
no_python_jobs=[]
for i in range(len(df_de_us_nona)):
    if 'python' in df_de_us_nona['job_skills'].iloc[i]:
        python_jobs.append(df_de_us_nona.index[i])
    else:
        no_python_jobs.append(df_de_us_nona.index[i])


df_de_us_python = df_de_us_nona[df_de_us_nona.index.isin(python_jobs)]

df_de_us_no_python = df_de_us_nona[~df_de_us_nona.index.isin(python_jobs)]





df_de_us_no_python = df_de_us_no_python.dropna(subset=['salary_year_avg'])
df_de_us_python = df_de_us_python.dropna(subset=['salary_year_avg'])

sns.set_theme(style='ticks')
sns.kdeplot(data=df_de_us_python, x='salary_year_avg' ,color='blue',label='Python Jobs' , alpha=0.5, fill=True)
sns.kdeplot(data=df_de_us_no_python, x='salary_year_avg',color='red',label='Non-Python Jobs' , alpha=0.5, fill=True)
plt.xlabel('Yearly Salary')
plt.ylabel('Density')
plt.title('Overlaid Salary Distributions of Data Engineer Jobs that Require Python vs those that do not.')
plt.legend()
ax=plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos:  f'${int(x/1000)}k'))

print(statistics.stdev(df_de_us_no_python['salary_year_avg']), statistics.stdev(df_da_us_python['salary_year_avg']))

t, p = stats.ttest_ind(df_de_us_python['salary_year_avg'], df_de_us_no_python['salary_year_avg'])

print(f"T-statistic: {t:.3f}")
print(f"P-value: {p:.10f}")
print(f"One-Sided P-Value:  {(p/2):.10f}")


```

### Results

![Overlaid Salary Distributions for Data Analyst Jobs that Require Python vs those that do not](3_project\images\da_dist.png)

![Overlaid Salary Distributions for Data Scientist Jobs that Require Python vs those that do not](3_project\images\ds_dist.png)

![Overlaid Salary Distributions for Data Engineer Jobs that Require Python vs those that do not](3_project\images\de_dist.png)

### Insights

1.  In all three cases, the advantage of learning Python was found to be statistically significant to the 0.005 level at least.

2.  I did use a generous interpretation of the equal variances assumption.  Salary data can be a bit left-skewed, in this case we have a few companies seeking highly-paid experts (of course most of those experts probably know Python).  A crude solution would be to use only the bottom 95% of salaries.

3.  I don't love the density scale on the Y-Axis, but I liked removing it even less.

4.  I'd be curious to repeat this step including the senior roles.  I'd expect the advantage might disappear, but I also wonder if there are enough senior roles that don't ask for Python.