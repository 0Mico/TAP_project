import requests
import json
import time
from bs4 import BeautifulSoup
from bs4.element import Tag
import urllib.parse

# Make an http get request to the url. Returns the response content
def _makeHTTPRequest(url: str):
    response = requests.get(url)
    return response.text

# Elaborate the response using BeautifulSoup's html parser
def _organizeResponse(response: str):
    soup = BeautifulSoup(response, "html.parser")
    return soup

# Extract the list of job cards in the given web page    
def _extractJobCardsFromHTML(web_page: BeautifulSoup):
    job_cards = web_page.select("li div.base-card")
    return job_cards

# Extract the job_id from the card received
def _extactJobIDFromHTML(job_card: Tag):
    job_id = job_card.get("data-entity-urn").split(":")[3]
    if job_id: return job_id
    else: return ''

# Use css selectors to extract the job title from the card received
def _extractTitleFromHTML(job_card: Tag):
    tag = job_card.select_one("a span")
    if tag:
        title = tag.get_text().strip()
        if title: return title
    else: return ''

# Extract the company name from the card received
def _extractCompanyNameFromHTML(job_card: Tag):
    tag = job_card.select_one("h4 a")
    if tag:
        company_name = tag.get_text().strip()
        if company_name: return company_name
    else: return ''

# Extract the location of the job from the card received
def _extractJobLocationFromHTML(job_card: Tag):
    tag = job_card.select_one("span.job-search-card__location")
    if tag:
        location = tag.get_text().strip()
        if location: return location
    else: return ''

# Extract the pubblication date of the job post
def _extractPubblicationDateFromHTML(job_card: Tag):
    tag = job_card.select_one("time")
    if tag:
        date = tag.get('datetime')
        if date: return date
    else: return ''

# Extract the link to go to the job page
def _goToJobPage(base_url: str, job_id: str):
    url = base_url + job_id
    response = _makeHTTPRequest(url)
    return response

# Extract the job description to retrieve then skills required
def _extractJobDescriptionFronHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("div.show-more-less-html__markup")
    if tag:
        description = tag.get_text().strip()
        if description: return description
    else: return ''

# Extract experience_level required for a job
def _extractSeniorityLevelfromHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("ul li:nth-child(1) span")
    if tag:
        seniority = tag.get_text().strip()
        if seniority: return seniority
    else: return ''

# Extract the contract type for a job
def _extractEmploymentTypeFromHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("ul li:nth-child(2) span")
    if tag:
        type = tag.get_text().strip()
        if type: return type
    else: return ''

# Extract job_function for a job
def _extractJobFunctionFromHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("ul li:nth-child(3) span")
    if tag:
        func = tag.get_text().strip()
        if func: return func
    else: return ''

# Extract indusrty type of the company that publiched a job post
def _extractIndustryTypeFromHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("ul li:nth-child(4) span")
    if tag:
        type = tag.get_text().strip()
        if type: return type
    else: return ''

# Update the url with the number of job_posting already scraped
def _modifyUrl(url: str, new_start: int):
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    query_params['start'] = [str(new_start)]
    new_query_string = urllib.parse.urlencode(query_params, doseq=True)
    new_url = parsed_url._replace(query=new_query_string).geturl()
    return new_url

# Create a JSON object for each job_card received
def _createJobObject(job_card: Tag):
    job = {}
    job['Job_ID'] = _extactJobIDFromHTML(job_card)
    job['Title'] = _extractTitleFromHTML(job_card)
    job['Company_name'] = _extractCompanyNameFromHTML(job_card)
    job['Location'] = _extractJobLocationFromHTML(job_card)
    job['Pubblication_date'] = _extractPubblicationDateFromHTML(job_card)

    response = _goToJobPage(job_link, job['Job_ID'])
    soup = _organizeResponse(response)
 
    job['Description'] = _extractJobDescriptionFronHTML(soup)
    job['Seniority_level'] = _extractSeniorityLevelfromHTML(soup)
    job['Employment_type'] = _extractEmploymentTypeFromHTML(soup)
    job['Job_Function'] = _extractJobFunctionFromHTML(soup)
    job['Industry_type'] = _extractIndustryTypeFromHTML(soup)
    return job

# Make a json object for each job scraped and write it into a json file   
def scrapeJobs(url: str, post_scraped: int):
    print(url) 
    response = _makeHTTPRequest(url)
    soup =_organizeResponse(response)
    job_cards = _extractJobCardsFromHTML(soup)
    jobs_retrieved = len(job_cards)
    
    for card in job_cards:
        job = _createJobObject(card)       
        
        with open("LinkedinJobPosts.json", "a") as file:
            json.dump(job, file, indent=4)
            file.write('\n')

    if jobs_retrieved > 0:
        post_scraped += jobs_retrieved
        new_url = _modifyUrl(url, post_scraped)
        
        #To not make the server reset the connection due to too much reqeusts in the unit of time
        time.sleep(1)

        scrapeJobs(new_url, post_scraped)

keywords = ['Data+Analyst', 'Data+Scientist', 'Cloud+Engineer', 'Devops', 'Frontend+Developer', 'Backend+Developer', 
            'Software+Engineer', 'Fullstack+Developer', 'Mobile+Developer', 'Game+Developer', 'Artificial+Intelligence']

job_link = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/"

for k in keywords:
    keyword = k
    start_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={k}&geoId=103350119&start=0"
    post_scraped = 0
    scrapeJobs(start_url, post_scraped)
