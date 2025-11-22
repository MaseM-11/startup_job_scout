'''
Takes a dataframe of jobs + my resume and a goal / preference doc, and rates the fit of each job (with GPT5)
'''
from exa_py import Exa
import pandas as pd
import numpy as np
import os
from openai import OpenAI
import re
import requests
import json
import asyncio
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
import gspread

load_dotenv()
exa_api_key = os.getenv("EXA_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
exa = Exa(api_key=exa_api_key)
openai = OpenAI(api_key=openai_api_key)

## Google Sheets API setups
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")
TAB_NAME = os.getenv("GOOGLE_SHEET_TAB_FITTED")
# Auth
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
google = gspread.authorize(creds)
# Open sheet and tab
sheet = google.open(SHEET_NAME).worksheet(TAB_NAME)

def remove_non_fits(job_title):
    '''
    Basic filter for jobs clearly not a fit, based on the job title, i.e. 'Manager', 'Senior', etc roles.
    '''
    wrong_seniority = ['Lead', 'Manager', 'Senior']

    pattern = re.compile(r'\b(?:' + '|'.join(wrong_seniority) + r')\b', re.IGNORECASE)

    if pattern.search(job_title):
        return True
    return False


def rate_job_fit(job_title, resume_text, goals, job_description=None):
    '''
    Takes in the job title, my resume and goal sheet, and can also also take in the description from the jobs page, and 
    returns a 1-10 rating of how much the job is a fit and a brief reason for why.
    '''
    prompt = f"""
    You are a career-matching assistant.
    Rate from 1–10 how well this job aligns with the candidate's background and goals.
    Be extremely discerning. These ratings will inform how I apply to jobs.
    I want to focus high efforts on a smaller number of jobs that are good fits.

    ### Candidate Resume
    {resume_text}

    ### Career Goals
    {goals}

    ### Job Posting
    {job_title}
    
    ### Job Description
    {job_description}

    Respond with only a JSON object:
    {{"score": <number>, "reason": "<EXTREMELY short explanation>"}}
    """

    resp = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    
    content = resp.choices[0].message.content
    if isinstance(content, list):
        content = " ".join([str(c) for c in content])

    try:
        result = json.loads(content)
        score = result.get("score")
        reason = result.get("reason")
    except json.JSONDecodeError:
        score, reason = None, content

    return (score, reason)


def summarize_job_text(text):
    '''
    Distills job description to a brief summary, listed responsibilies, requirements, and years of experience requested.

    text (str): job description scraped from job listing
    '''

    prompt = f"""
    Summarize the following job posting text.
    Return a **valid JSON object** with this EXACT structure:
    {{
      "summary": "Brief overview of the role and company",
      "responsibilities": ["Key responsibilities"],
      "requirements": ["List of requirements"],
      "years_experience_required": <integer or null>
    }}

    --- Job Posting ---
    {text}
    """

    resp = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    content = resp.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "summary": None,
            "responsibilities": [],
            "requirements": [],
            "years_experience_required": None,
        }

    return result


def process_fit(row, df, resume_text, goals):
    '''
    For a given company (row), handles the calls to the filter/remove and rating functions.
    '''
    try:
        job = df.loc[row, 'title']

        if remove_non_fits(job):
            return {
                "index": row,
                "fit": 0,
                "reason": "Seniority"
            }

        score, reason = rate_job_fit(job, resume_text, goals, None)
        return {
            "index": row,
            "fit": score,
            "reason": reason
        }
    except Exception as e:
        print(f"⚠️ Error processing row {row}: {e}")
        return {
            "index": row,
            "fit": None,
            "reason": "Error"
        }


def process_row_with_desc(row, df, resume_text, goals):
        try:
            title = df.loc[row, 'title']
            url = df.loc[row, 'link']
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = r.text[:10000]
            cleaned_text = re.sub(r"<.*?>", "", text)

            result = summarize_job_text(cleaned_text)
            summary = result['summary']
            resp = result['responsibilities']
            req = result['requirements']
            yrs_req = result['years_experience_required']
            desc = f"Summary:\n{summary}\nResponsibilities:\n{resp}\nRequirements:\n{req}"

            if (not yrs_req) or (yrs_req < 4):
                final_fit, reason = rate_job_fit(title, resume_text, goals, desc)
            else:
                final_fit, reason = 0, 'Seniority'

            return {
                "index": row,
                "summary": summary,
                "resp": resp,
                "req": req,
                "yrs_req": yrs_req,
                "final_fit": final_fit,
                "reason": reason
            }
        except Exception as e:
            print(f"⚠️ Error processing row {row}: {e}")
            return {
                "index": row,
                "summary": None,
                "resp": [],
                "req": [],
                "yrs_req": None,
                "final_fit": 0,
                "reason": "Error"
            }
        

## ---------------------------------------------------------------- ##
    

def main():
    input_file = sys.argv[1]
    test_mode = "--test" in sys.argv
    base_name = Path(input_file).stem
    df_final = pd.read_csv(f'scraped_data/jobs_scraped_{base_name}.csv')
    if test_mode:
        df_final = df_final.head(3)
    ## FIXME need to read in the dataframe from prev script

    ## Can adjust for how many parrallel threads running at a time.
    num_threads=10

    ## Read in my resume and goals/preference doc.
    resume_text = open("input_personal_info/MasonMangum.txt").read()
    goals = open("input_personal_info/GoalStatement.txt").read()

    ## First pass filter to remove obvious non-fit jobs by job title.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_fit, i, df_final, resume_text, goals) for i in range(len(df_final))]
        for future in as_completed(futures):
            result = future.result()
            i = result["index"]
            df_final.loc[i, "fit"] = result["fit"]
            df_final.loc[i, "fit reason"] = result["reason"]

    ## Remove all jobs where initial fit rating is less than 5.
    df_final = df_final.reset_index(drop=True)
    df_with_fits = df_final[df_final['fit'] > 4].reset_index(drop=True)

    ## Add columns to the dataframe that will be added on the second pass.
    df_with_fits_scraped = df_with_fits.copy()
    for col in ['role description', 'role responsibilities', 'role reqs', 'role years experience', 'final fit', 'fit reason']:
        if col not in df_with_fits_scraped.columns:
            df_with_fits_scraped[col] = None

    # Second pass with the job description included. Run in parrallel.
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_row_with_desc, i, df_with_fits_scraped, resume_text, goals) for i in range(len(df_with_fits_scraped))]
        for future in as_completed(futures):
            result = future.result()
            i = result["index"]
            df_with_fits_scraped.loc[i, 'role description'] = result["summary"]
            df_with_fits_scraped.at[i, 'role responsibilities'] = result["resp"]
            df_with_fits_scraped.at[i, 'role reqs'] = result["req"]
            df_with_fits_scraped.loc[i, 'yrs exp req'] = result["yrs_req"]
            df_with_fits_scraped.loc[i, 'final fit'] = result["final_fit"]
            df_with_fits_scraped.loc[i, 'fit reason'] = result["reason"]

    ## FIXME need to save the dataframe for the next script
    ## FIXME change it so that it saves the name for whatever scrape you wanna call it
    if test_mode:
        print(df_with_fits_scraped['title'])
    else:
        df_with_fits_scraped.to_csv(f"scraped_data/jobs_scraped_fitted_{input_file}.csv", index=False)
        df_cleaned = (df_with_fits_scraped.replace([float('inf'), float('-inf')], None)
                            .fillna("")
                            .infer_objects(copy=False)
                        )
        df_cleaned = df_cleaned.applymap(
            lambda x: "\n".join(x) if isinstance(x, list) else str(x)
        )
        sheet.append_rows(
            df_cleaned.values.tolist(),
            value_input_option="USER_ENTERED",
            table_range="A1"
        )
        print("✅ Data appended to Google Sheet!")
    

if __name__ == "__main__":
    main()

