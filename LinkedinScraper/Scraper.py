from hmac import new
import requests
import json
import time
from bs4 import BeautifulSoup
from bs4.element import Tag
import urllib.parse
import random

# Make an http get request to the url. Returns the response content
def _makeHTTPRequest(url: str, max_retries: int = 5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 5 + random.uniform(0, 5)  # Exponential backoff
                print(f"Rate limited (429). Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            response.raise_for_status() # Raise error for a 500 status (the script would treat it as a success without this line)
            return response.text
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    print(f"Failed to fetch {url} after {max_retries} attempts")
    return None

# Elaborate the response using BeautifulSoup's html parser
def _organizeResponse(response: str):
    if response is None:
        return None
    soup = BeautifulSoup(response, "html.parser")
    return soup

# Extract the list of job cards in the given web page    
def _extractJobCardsFromHTML(web_page: BeautifulSoup):
    if web_page is None:
        return []
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
# Extract text recursively from the "container tag" that contains the entire description
def _extractJobDescriptionFronHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("div.show-more-less-html__markup")
    if tag:
        description = tag.get_text().strip()
        if description == "":
            print("searching in child tags")
            for child in tag.descendants:
                description = description.join(child.get_text().strip())
        return description
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

# Send the job to the logstash container
def send_to_logstash(job):
    try:
        url = 'http://logstash:5000/'
        headers = {'Content-Type': 'application/json'}
        requests.post(url, json=job, headers=headers)
    except Exception as e:
        print(f"Error sending job to logstash: {e}")

# Make a json object for each job scraped and write it into a json file   
def scrapeJobs(url: str):
    post_scraped = 0
    while True:
        print(url) 
        response = _makeHTTPRequest(url)
        if response is None:
            print("Failed to retrieve data, stopping scrape.")
            break

        soup =_organizeResponse(response)
        if soup is None:
            print("Failed to parse data, stopping scrape.")
            break

        job_cards = _extractJobCardsFromHTML(soup)
        jobs_retrieved = len(job_cards)
        if jobs_retrieved == 0:
            print("No more jobs to scrape, stopping scrape for this keyword.")
            break
        
        for card in job_cards:
            job = _createJobObject(card)

            print(json.dumps(job, indent=4))

            send_to_logstash(job)
            
            '''with open("LinkedinJobPosts.json", "a") as file:
                json.dump(job, file, indent=4)
                file.write('\n')'''

        post_scraped += jobs_retrieved
        new_url = _modifyUrl(url, post_scraped)
        url = new_url
            
        #To not make the server reset the connection due to too much requests in the unit of time
        time.sleep(random.uniform(2, 4))


if __name__ == "__main__":
    keywords = ['Data+Analyst', 'Frontend+Developer', 'Data+Scientist', 'Cloud+Engineer', 'Backend+Developer', 'Devops',  
                'Software+Engineer', 'Fullstack+Developer', 'Mobile+Developer', 'Game+Developer', 'Artificial+Intelligence',
                'Python+Developer']

    job_link = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/"
    for k in keywords:
        start_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={k}&geoId=103350119&start=0"
        try:
            scrapeJobs(start_url)  
        except Exception as e:
            print(f"Fatal error scraping {k}: {e}")
            continue