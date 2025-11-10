# Job Scrape Scout

This project scrapes startup jobs, filters them against your skillset and goals, then launches an interactive demo for accelerated applying.

The goal of this project is to make sure I am targeting the optimal subset of all applicable startup jobs.
   - Aggregate all startup jobs --> leave no stone uncovered, explore all possible options
   - Filter by hard capped seniority requirements --> seniority mismatches are demoralizing and waste lots of time
   - Rate and sort them by skillset / interest match --> spend more time on less (more exciting) applications

## Workflow
run_all.py takes in company names and brief descriptions, and runs the whole workflow, ending with links to apply.

scrape_jobs.py -> rate_job_fit.py -> application_workflow.py

### Data storage files
   - input_companies: stores .txt files of the lists of companies that you want scraped
   - scraped_data: stores .csv files (pandas dataframes) of scraped jobs from scrape_jobs.py and rate_job_fit.py
   - apply_data: stores the jobs you checkbox 'interested' in and jobs you checkbox that you've applied to for 'application_workflow.py' demo

### scrape_jobs.py
Given company names, uses Exa to find the companies URL and startups gallery profile. Then uses Exa to scrape the gallery page for open roles.

Input: .txt file of company names + ~5 word descriptions
   - Run with 'python scrape_jobs.py "{companies_list}.txt"'
Output: .csv file where each row is a job
   - Stored in scraped_data as "jobs_scraped_{companies_list}.csv"
   - Includes columns ['Company, Description, URL, title, location, date_posted, link']

### rate_job_fit.py
Given the scraped job roles and a resume + goal / preferences doc, this file uses GPT to rate the fit of each job. 

The workflow looks like this:
1. Filters obvious non-fits
   Starting with the list of roles, it first filters through obvious seniority mismatches (i.e. manager roles). Then using GPT, rates the job name's fit to candidates skillset on a 1-10 scale, and filters those whose match is 4 or less.
2. Deeper rate and sort
   For remaining jobs, it generates a summary for each role and matches that summary to a resume + goal statement, outputting a 1-10 scale fit rating. Finally, sort the jobs best fit to worst fit.

Input: .csv file of jobs (as outputted by scrape_jobs.py)
   - Run with 'python rate_job_fit.py "jobs_scraped_{companies_list}.csv"'
Output: .csv file of filtered and sorted jobs
   - Stored in scraped_data as "jobs_scraped_fitted_{companies_list}.csv"
   - Includes columns [Company, Description, URL, title, location, date_posted, link, fit, fit reason, role description, role responsibilities, role reqs, role years experience, final fit, yrs exp req] --> sorted by 'final fit'

### application_workflow.py
Given the list of roles, sorted by fit rating and enriched with scraped data / GPT summaries, launches a Gradio UI. The app consists of two tabs, 'Interested Jobs' and 'Applied Jobs'. Interested jobs is a checklist of the inputted jobs, listed with -- Name of the role, location, posted date, and fit/rating. When you check that you're "Interested" in one of the jobs, it live updates / adds it to the 'Applied Jobs' tab, which is also a checklist, this time included more info including a summary, responsibilies, requirements, and the link to apply.


