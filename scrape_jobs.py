''' 
Takes a str of companies and their descriptions, and loops through them to scrape open jobs.

Input: txt file formatted as company_name\ncompany_description\n\n restart

Output: 
'''

from exa_py import Exa
import pandas as pd
import numpy as np
import os
from openai import OpenAI
import re
import requests
import ast
import json
from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path


EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise RuntimeError("EXA_API_KEY environment variable is not set.")

exa = Exa(api_key=EXA_API_KEY)


def get_company_url(company_name, description):
    ''' 
    Uses Exa API. Takes in a company name and description, then outputs that company's homepage url. 
    '''
    q = f'{company_name} {description} site:.com OR site:.io OR site:.ai OR site:.co'
    res = exa.search(
        query=q,
        category="company",
        num_results=5
    )
    if not res.results:
        return None
    
    for r in res.results:
        if company_name.lower().split()[0] in r.url.lower():
            return r.url
    return res.results[0].url


def get_job_urls(company_name):
    '''
    Uses Exa API. Takes in a company's name, finds its profile on startups gallery, and finds the link to its job page.
    '''
    slug = company_name.lower().replace(" ", "-")
    url = f"https://startups.gallery/companies/{slug}"
    
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        if r.status_code >= 400:
            print(f"⚠️ No gallery page for: {company_name} ({url})")
            return None
    except requests.RequestException:
        print(f"⚠️ Connection error for: {company_name} ({url})")
        return None
    
    result = exa.get_contents(
      [url],
      text = {
        "max_characters": 500
      },
      livecrawl = "preferred",
      summary={
            "query": "Find the link for their jobs page (the 'View Jobs' link).",
            "schema": {
                "type": "object",
                "properties": {
                    "job_page_url": {"type": "string"}
                }
            }
       }
    )
    
    if not result.results:
        print(f"⚠️ No jobs found for: {company_name} ({url})")
        return []
    
    summary = result.results[0].summary
    print(summary)
    summary = json.loads(summary)
    link = summary.get("job_page_url")

    return link


def get_jobs(company_name):
    '''
    Uses Exa API. Takes in a companies name, finds its profile and scrapes all jobs listed on startup gallery.
    Prints exception statements for any company who's profile couldn't be found, has no jobs listed, etc.
    '''

    slug = company_name.lower().replace(" ", "-").replace(".", "-")
    url = f"https://startups.gallery/companies/{slug}"
    
    try:
        r = requests.head(url, allow_redirects=True, timeout=5)
        if r.status_code >= 400:
            print(f"⚠️ No gallery page for: {company_name} ({url})")
            return [], None
    except requests.RequestException:
        print(f"⚠️ Connection error for: {company_name} ({url})")
        return [], None
    
    result = exa.get_contents(
      [url],
      text = True,
      summary={
        "query": (
            "Find all jobs listed and include their titles, locations, dates, and links. "
            "Also report the company's fundraising stage if it is mentioned."
        ),
        "schema": {
            "description": "Schema describing a collection of job listings",
            "type": "object",
            "properties": {
                "fundraising_stage": {
                    "type": "string",
                    "description": "Company fundraising stage or round"
                },
                "jobs": {
                    "type": "array",
                    "description": "List of job listings",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "location": {"type": "string"},
                            "date_posted": {"type": "string"},
                            "link": {"type": "string"}
                        }
                    }
                }
            }
            }
            }
        )
    
    if not result.results:
        print(f"⚠️ No jobs found for: {company_name} ({url})")
        return [], None

    summary = result.results[0].summary
    try:
        parsed_summary = json.loads(summary)
        jobs = parsed_summary.get('jobs', [])
        fundraising_stage = parsed_summary.get('fundraising_stage')
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON for {company_name}")
        return [], None

    if jobs:
        if jobs[0]['title'] == 'View Jobs':
            print('-------------------------')
            return [], None
    return jobs, fundraising_stage


def process_company(row):
    '''
    For a given company (row), handles the calls to Exa url finder and jobs finder functions.
    '''
    company_name = row["Company"]
    description = row["Description"]
    url = get_company_url(company_name, description)
    jobs, fundraising_stage = get_jobs(company_name)
    return (company_name, url, jobs, fundraising_stage)


## ---------------------------------------------------------------- ##


def main():

    input_file = sys.argv[1]
    test_mode = "--test" in sys.argv

    gallery_list = open(f'input_companies/{input_file}').read()

    entries = [block.strip() for block in re.split(r"\n{2,}", gallery_list) if block.strip()]
    data = [entry.split("\n", 1) for entry in entries]
    companies = pd.DataFrame(data, columns=["Company", "Description"])

    ## For testing purposes -- can adjust how many companies are processed per run, and how many processed in parrallel
    if test_mode:
        print("🧪 Running in TEST MODE (processing only 3 companies)")
        test_df = companies.head(3)
        num_threads = 2
    else:
        test_df = companies
        num_threads = 10

    test_df_urls = test_df.copy()
    test_df_urls["URL"] = None
    test_df_urls["Jobs"] = None
    test_df_urls["Fundraising Stage"] = None

    ## Run (num_threads) parrallel calls to Exa
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_company, row): i for i, row in test_df.iterrows()}
        
        ## Loops through all companies and adds their url / jobs to a dataframe
        for future in as_completed(futures):
            i = futures[future]
            try:
                company_name, url, job_list, fundraising_stage = future.result()
                test_df_urls.at[i, "URL"] = url
                test_df_urls.at[i, "Jobs"] = job_list
                test_df_urls.at[i, "Fundraising Stage"] = fundraising_stage or "Unknown"
            except Exception as e:
                print(f"⚠️ Error processing company at index {i}: {e}")
                test_df_urls.at[i, "URL"] = None
                test_df_urls.at[i, "Jobs"] = []
                test_df_urls.at[i, "Fundraising Stage"] = "Unknown"

    ## Expands the dataframe from one row per company, to one row per job.
    df_expanded = test_df_urls.explode("Jobs", ignore_index=True)
    df_expanded = df_expanded[df_expanded["Jobs"].notna()].reset_index(drop=True)

    if df_expanded.empty:
        df_final = pd.DataFrame(columns=list(test_df_urls.columns) + [
            "title",
            "location",
            "date_posted",
            "link",
        ])
    else:
        job_details = pd.json_normalize(df_expanded["Jobs"])
        df_final = pd.concat(
            [
                df_expanded.drop(columns=["Jobs"]).reset_index(drop=True),
                job_details.reset_index(drop=True),
            ],
            axis=1,
        )

    for col in ["title", "location", "date_posted", "link", "fit", "fit reason"]:
        if col not in df_final.columns:
            df_final[col] = None

    if "Fundraising Stage" not in df_final.columns:
        df_final["Fundraising Stage"] = "Unknown"
    else:
        df_final.loc[:, "Fundraising Stage"] = df_final["Fundraising Stage"].fillna("Unknown")

    df_final = df_final[df_final["title"].notna()].reset_index(drop=True)

    ## FIXME need to save the final dataframe so can use in the other scripts
    if test_mode:
        print(df_final['title'])
    base_name = Path(input_file).stem
    df_final.to_csv(f"scraped_data/jobs_scraped_{base_name}.csv", index=False)

if __name__ == "__main__":
    main()
