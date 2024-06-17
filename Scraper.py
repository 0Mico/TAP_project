import requests
import json
import time
from bs4 import BeautifulSoup
from bs4.element import Tag

# Make an http get request to the url. Returns the response content
def _makeHTTPRequest(url: str):
    response = requests.get(url)
    return response.text

# Elaborate the response using BeautifulSoup's html parser
def _organizeResponse(response: str):
    soup = BeautifulSoup(response, "html.parser")
    """ with open("output.html", "w") as f:
        f.write(soup.prettify()) """
    return soup

# Extract the list of job cards in the given web page    
def _extractJobCardsFromHTML(web_page: BeautifulSoup):
    job_cards = web_page.select("li div.base-card")
    return job_cards

# Extract the job_id from the card received
def _extactJobIDFromHTML(job_card: Tag):
    job_id = job_card.get("data-entity-urn").split(":")[3]
    return job_id

# Use css selectors to extract the job title from the card received
def _extractTitleFromHTML(job_card: Tag):
    tag = job_card.select_one("a span")
    title = tag.get_text().strip()
    return title

# Extract the company name from the card received
def _extractCompanyNameFromHTML(job_card: Tag):
    tag = job_card.select_one("h4 a")
    company_name = tag.get_text().strip()
    return company_name

# Extract the location of the job from the card received
def _extractJobLocationFromHTML(job_card: Tag):
    tag = job_card.select_one("span.job-search-card__location")
    location = tag.get_text().strip()
    return location

# Extract the pubblication date of the job post
def _extractPubblicationDateFromHTML(job_card: Tag):
    tag = job_card.select_one("time")
    date = tag.get("datetime")
    return date

# Extract the link to go to the job page
def _extractJobLinkFromHTML(job_card: Tag):
    tag = job_card.select_one("a.base-card__full-link")
    url = tag.get("href")
    return url

# Extract the job description to retrieve then skills required
def _extractJobDescriptionFronHTML(web_page: BeautifulSoup):
    tag = web_page.select_one("div.show-more-less-html__markup")
    if tag:
        description = tag.get_text().strip()
        if description:
            return description
        else:
            return ''
    else:
        return ''

# Extract expereince_level, contract_type, area and industry_type
def _extractExperienceLevelFromHTML(web_page: BeautifulSoup):
    span_tags = web_page.select("span.description__job-criteria-text")
    infos = []
    if span_tags:
        for tag in span_tags:
            infos.append(tag.get_text().strip())
    return infos

# Make a json object for each job scraped and write it into a json file   
def scrapeJobs(url):
    response = _makeHTTPRequest(url)
    soup =_organizeResponse(response)
    job_cards = _extractJobCardsFromHTML(soup)
    jobs = []

    for card in job_cards:
        job = {}
        job['Job_ID'] = _extactJobIDFromHTML(card)
        job['Title'] = _extractTitleFromHTML(card)
        job['Company_name'] = _extractCompanyNameFromHTML(card)
        job['Location'] = _extractJobLocationFromHTML(card)
        job['Pubblication_date'] = _extractPubblicationDateFromHTML(card)

        url = _extractJobLinkFromHTML(card)
        print(url)
        time.sleep(10)
        response = _makeHTTPRequest(url)
        soup = BeautifulSoup(response, "html.parser")
        """ with open("job_page.html", "w") as f:
            f.write(soup.prettify()) """   
        job['Description'] = _extractJobDescriptionFronHTML(soup)
        infos = _extractExperienceLevelFromHTML(soup)
        print(infos)
        if infos:
            job['Experience_level'] = infos[0] if infos[0] else ''
            job['Contract_type'] = infos[1] if infos[1] else ''
            job['Area'] = infos[2] if infos[2] else ''
            job['Industry_type'] = infos[3] if infos[3] else ''
        
        jobs.append(job)
        
        with open("LinkedinJobPosts.json", "a") as file:
            json.dump(job, file, indent=4)
            file.write('\n')
   
url = "https://www.linkedin.com/jobs/search/?geoId=103350119&keywords=C++"
scrapeJobs(url)

